"""Run the analysis pipeline on a video, or on one section of it.

    python main.py                                       # default video, in full
    python main.py input_videos/game.mp4 --random-clip   # random 30s section
    python main.py input_videos/game.mp4 --random-clip --clip-duration 60 --seed 5
    python main.py input_videos/game.mp4 --start 12:30 --end 13:00

Detections are cached in stubs/ keyed by video name and section, so running
the same section again (same --seed, or the same --start/--end) skips
inference. Pass --no-cache to force a fresh run.
"""
import argparse
import math
import os
import random
import sys

import numpy as np

from utils import (read_video, save_video, probe_video, parse_timestamp,
                   format_timestamp, resolve_window, TimestampError)
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import pitch_calibration
from pitch_calibration import CalibratedTransformer
from pitch_view import PitchRenderer, inset
from utils import get_center_of_bbox

DEFAULT_VIDEO = './input_videos/yt_download.f398.mp4'
# The bundled stubs were generated from this clip and only line up with it.
SAMPLE_VIDEO = 'input_videos/08fd33_4.mp4'
SAMPLE_STUBS = ('stubs/track_stubs.pkl', 'stubs/camera_movement_stub.pkl')
# Kit colours are clustered from the shirts, but a kit close to grass
# (Mexico's green here) averages out to a washed grey-green that is
# nearly the colour of the other side once drawn. Cluster on the real
# colours, draw with these.
TEAM_DISPLAY = {1: (235, 130, 40), 2: (55, 55, 235)}   # BGR: blue, red
# Raw frames above this are refused without a section (output doubles it).
MAX_FULL_VIDEO_GB = 8


def timestamp_arg(text):
    try:
        return parse_timestamp(text)
    except TimestampError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the football analysis pipeline on a video or a section of it.")
    parser.add_argument('video', nargs='?', default=DEFAULT_VIDEO,
                        help=f"input video (default: {DEFAULT_VIDEO})")
    parser.add_argument('--start', type=timestamp_arg, default=None,
                        help="section start, e.g. 90, 1:30 or 00:01:30")
    parser.add_argument('--end', type=timestamp_arg, default=None,
                        help="section end; defaults to start + --clip-duration")
    parser.add_argument('--random-clip', action='store_true',
                        help="analyse a random section instead of naming times")
    parser.add_argument('--clip-duration', type=float, default=30.0,
                        help="section length in seconds (default: 30)")
    parser.add_argument('--seed', type=int, default=None,
                        help="seed for --random-clip, to repeat a selection")
    parser.add_argument('--model', default='models/best.pt')
    parser.add_argument('--output', default=None,
                        help="output path (default: output_videos/<video>_<start>-<end>.avi)")
    parser.add_argument('--device', default=None,
                        help="inference device: 0 for the first GPU, 'cpu', or omit to autodetect")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="inference resolution; 1280 finds the ball far more reliably "
                             "but costs about 5x on CPU")
    parser.add_argument('--half', action='store_true',
                        help="half precision, GPU only (roughly 2x)")
    parser.add_argument('--batch', type=int, default=20,
                        help="frames per inference batch; raise it on a GPU")
    parser.add_argument('--no-cache', action='store_true',
                        help="ignore cached detections in stubs/ and run inference again")
    parser.add_argument('--no-pitch-view', action='store_true',
                        help="skip the 2D pitch inset")
    parser.add_argument('--no-track-calibration', action='store_true',
                        help="use one fixed calibration instead of following the camera")
    parser.add_argument('--no-team-shape', action='store_true',
                        help="plot players only, without hull and centroid")
    parser.add_argument('--kit-colours', action='store_true',
                        help="draw teams in their detected kit colour instead of "
                             "the fixed high-contrast palette")
    parser.add_argument('--pitch-view-scale', type=float, default=0.34,
                        help="width of the pitch inset as a fraction of the frame")
    return parser.parse_args()


def section_label(video, window):
    stem = os.path.splitext(os.path.basename(video))[0]
    if window is None:
        return stem
    return f"{stem}_{window[0]}-{window[1]}"


def stub_paths(video, window, imgsz=640):
    """Cache files for this exact video, section and inference size.

    imgsz is part of the key because detections differ substantially with it:
    the ball is found in roughly a third of frames at 640 and nearly all of
    them at 1280. Without it in the name, switching resolution would silently
    reuse the old, worse detections.
    """
    if window is None and os.path.abspath(video) == os.path.abspath(SAMPLE_VIDEO):
        return SAMPLE_STUBS
    label = section_label(video, window)
    suffix = '' if imgsz == 640 else f'_{imgsz}'
    return (f'stubs/{label}{suffix}_tracks.pkl',
            f'stubs/{label}{suffix}_camera.pkl')


def first_frame_with_players(player_tracks, minimum=2):
    """Team colours are clustered from one frame, so it needs players in it."""
    for frame_num, players in enumerate(player_tracks):
        if len(players) >= minimum:
            return frame_num
    return None


def add_pitch_view(frames, tracks, calibration, scale=0.34, team_shape=True,
                   tracked=None):
    """Composite a bird's-eye view of the tracked positions onto each frame."""
    renderer = PitchRenderer(scale=6)
    for frame_num, frame in enumerate(frames):
        panel = renderer.blank()
        teams = {}
        for info in tracks['players'][frame_num].values():
            pos = info.get('position_transformed')
            if pos is None:
                continue
            team = info.get('team', 0)
            entry = teams.setdefault(
                team, {'pts': [], 'colour': info.get('team_color', (230, 230, 230))})
            entry['pts'].append(pos)
        for team in sorted(teams):
            pts, colour = teams[team]['pts'], teams[team]['colour']
            if team_shape:
                renderer.draw_shape(panel, pts, colour)
            renderer.draw_players(panel, pts, colour)
        # The ball is re-interpolated after the transform step, so its pitch
        # position has to be derived here from the (possibly filled) bbox.
        bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
        if bbox and np.all(np.isfinite(bbox)):
            cal = (tracked[min(frame_num, len(tracked) - 1)]
                   if tracked else calibration)
            centre = np.array([get_center_of_bbox(bbox)], dtype=np.float32)
            pitch_pt = cal.to_pitch(centre)
            if cal.on_pitch(pitch_pt)[0]:
                renderer.draw_ball(panel, pitch_pt)
        inset(frame, panel, scale=scale)
    return frames


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")
    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model} (the README links the download)")

    info = probe_video(args.video)
    try:
        window = resolve_window(info['duration'], args.start, args.end,
                                args.random_clip, args.clip_duration)
    except TimestampError as exc:
        sys.exit(str(exc))
    if window is not None:
        # Whole seconds keep the cache label exact for a repeated section.
        window = (math.floor(window[0]),
                  min(math.ceil(window[1]), math.ceil(info['duration'])))

    if window is None:
        frames_gb = info['frame_count'] * info['width'] * info['height'] * 3 / 1e9
        if frames_gb > MAX_FULL_VIDEO_GB:
            sys.exit(f"Reading the whole video needs about {frames_gb:.0f} GB of memory. "
                     "Use --random-clip or --start/--end to analyse a section.")

    fps = info['fps']
    start, end = window if window else (None, None)
    track_stub, camera_stub = stub_paths(args.video, window, args.imgsz)
    output_path = args.output or f"output_videos/{section_label(args.video, window)}.avi"
    os.makedirs('stubs', exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    use_cache = not args.no_cache

    print(f"Video: {args.video}  {info['width']}x{info['height']} @ {fps:.2f} fps, "
          f"{format_timestamp(info['duration'])} long")
    if window:
        print(f"Section: {format_timestamp(start)} to {format_timestamp(end)} ({end - start}s)")
    if use_cache and os.path.exists(track_stub):
        print(f"Using cached detections: {track_stub}")

    # Read Video
    video_frames = read_video(args.video, start, end)
    if not video_frames:
        sys.exit("No frames could be read from that section.")
    print(f"Loaded {len(video_frames)} frames")

    # Initialize Tracker
    tracker = Tracker(args.model, device=args.device, imgsz=args.imgsz,
                      half=args.half, batch_size=args.batch)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=use_cache,
                                       stub_path=track_stub)
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=use_cache,
                                                                                stub_path=camera_stub)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View transformer: a fitted homography for this clip when we have one.
    calibration = pitch_calibration.load(args.video, window)
    if calibration is not None:
        print(f"Calibration: {calibration.error:.2f}px mean line alignment")
        if args.no_track_calibration:
            transformer = CalibratedTransformer(calibration)
        else:
            # A pan changes perspective, which no translation can undo, so
            # re-fit periodically through the clip rather than trusting the
            # opening frame all the way to the end.
            tracked = pitch_calibration.track(video_frames, calibration)
            errs = [c.error for c in tracked if c.error is not None]
            if errs:
                print(f"Tracked calibration across the clip: "
                      f"{np.mean(errs):.2f}px mean, {max(errs):.2f}px worst")
            calibration = tracked[0]
            transformer = CalibratedTransformer(tracked)
    else:
        print("No calibration for this clip. Speed, distance and the pitch "
              "view need one - without it ViewTransformer's built-in "
              "coordinates apply, which belong to a different camera.\n"
              f'  python -m pitch_calibration.interactive "{args.video}"'
              + (f' --start {format_timestamp(start)} --end {format_timestamp(end)}'
                 if window else ''))
        transformer = ViewTransformer()
    transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    colour_frame = first_frame_with_players(tracks['players'])
    if colour_frame is None:
        sys.exit("No frame with at least two players detected; nothing to analyse. "
                 "Try another section, or --no-cache if the cached detections are stale.")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[colour_frame],
                                    tracks['players'][colour_frame])

    if not args.kit_colours:
        for team, drawn in TEAM_DISPLAY.items():
            kit = np.round(team_assigner.team_colors[team]).astype(int).tolist()
            name = {1: 'blue', 2: 'red'}[team]
            print(f"  team {team} drawn {name}; detected kit colour BGR {kit}")

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = (
                team_assigner.team_colors[team] if args.kit_colours
                else TEAM_DISPLAY.get(team, (230, 230, 230)))


    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # 0 = nobody yet; the overlay shows 0% until the first possession.
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control= np.array(team_ball_control)


    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,
                                                   team_ball_control,
                                                   in_place=True)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame, in_place=True)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    ## Draw the 2D pitch view
    if calibration is not None and not args.no_pitch_view:
        output_video_frames = add_pitch_view(
            output_video_frames, tracks, calibration,
            scale=args.pitch_view_scale, team_shape=not args.no_team_shape,
            tracked=None if args.no_track_calibration else tracked)

    # Save video
    save_video(output_video_frames, output_path, fps=fps)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
