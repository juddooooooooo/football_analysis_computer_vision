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
import pickle
import random
import sys

import numpy as np

from utils import (read_video, save_video, probe_video, parse_timestamp,
                   format_timestamp, resolve_window, TimestampError,
                   iter_video, count_frames, open_writer)
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import pitch_calibration
from pitch_calibration import CalibratedTransformer
from pitch_view import PitchRenderer, inset
from ball_tracking import BallTracker
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
CALIBRATION_EVERY = 10   # frames between calibration refits


def available_memory_gb():
    """Free RAM, or None when it cannot be determined on this platform."""
    try:                                  # Linux, so Colab
        import os as _os
        return (_os.sysconf('SC_AVPHYS_PAGES') *
                _os.sysconf('SC_PAGE_SIZE') / 1e9)
    except (ValueError, AttributeError, OSError):
        pass
    try:                                  # Windows
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1e9
    except Exception:
        return None


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
    parser.add_argument('--chunk', type=int, default=90,
                        help="frames held in memory at once; lower it if memory is tight")
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
                   tracked=None, renderer=None):
    """Composite a bird's-eye view of the tracked positions onto each frame."""
    renderer = renderer or PitchRenderer(scale=6)
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

    fps = info['fps']
    start, end = window if window else (None, None)
    n_expected = count_frames(args.video, start, end)
    track_stub, camera_stub = stub_paths(args.video, window, args.imgsz)
    output_path = args.output or f"output_videos/{section_label(args.video, window)}.avi"
    os.makedirs('stubs', exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    use_cache = not args.no_cache

    print(f"Video: {args.video}  {info['width']}x{info['height']} @ {fps:.2f} fps, "
          f"{format_timestamp(info['duration'])} long")
    if window:
        print(f"Section: {format_timestamp(start)} to {format_timestamp(end)} ({end - start}s)")
    if n_expected <= 0:
        sys.exit("That section contains no frames.")
    per_frame = info['width'] * info['height'] * 3 / 1e9
    print(f"{n_expected} frames, streamed {args.chunk} at a time: "
          f"{args.chunk * per_frame:.2f} GB held rather than "
          f"{n_expected * per_frame:.1f} GB")

    tracker = Tracker(args.model, device=args.device, imgsz=args.imgsz,
                      half=args.half, batch_size=args.batch)
    calibration = pitch_calibration.load(args.video, window)

    cached_tracks = use_cache and os.path.exists(track_stub)
    cached_camera = use_cache and os.path.exists(camera_stub)
    if cached_tracks:
        print(f"Using cached detections: {track_stub}")
        with open(track_stub, 'rb') as handle:
            tracks = pickle.load(handle)
    else:
        tracks = tracker.new_tracks()
    camera_movement = None
    if cached_camera:
        with open(camera_stub, 'rb') as handle:
            camera_movement = pickle.load(handle)

    if calibration is not None:
        print(f"Calibration: {calibration.error:.2f}px mean line alignment")
    else:
        print("No calibration for this clip. Speed, distance, the pitch view "
              "and the ball filter all need one - without it ViewTransformer's "
              "built-in coordinates apply, which belong to a different camera.\n"
              f'  python -m pitch_calibration.interactive "{args.video}"'
              + (f' --start {format_timestamp(start)} --end {format_timestamp(end)}'
                 if window else ''))

    # ---- First pass: everything that has to look at the frames.
    # Detection, camera movement, calibration refits and kit colours are all
    # gathered in one sweep, so the clip is decoded twice in total rather
    # than being held in memory for its whole length.
    follow = calibration is not None and not args.no_track_calibration
    estimator = None
    calib_keys, last_key = {}, None
    team_assigner = TeamAssigner()
    team_ready = False
    seen = 0

    for chunk in iter_video(args.video, start, end, chunk=args.chunk):
        if estimator is None:
            estimator = CameraMovementEstimator(chunk[0])
            if camera_movement is None:
                estimator.begin_stream(chunk[0].shape[1], chunk[0].shape[0])
        if not cached_tracks:
            tracker.track_chunk(chunk, tracks)
        if camera_movement is None:
            estimator.feed(chunk)

        for offset, frame in enumerate(chunk):
            index = seen + offset
            if index >= len(tracks['players']):
                break
            if follow and index % CALIBRATION_EVERY == 0:
                seed = calib_keys[last_key] if last_key is not None else calibration
                calib_keys[index] = pitch_calibration.refine_from(frame, seed)
                last_key = index
            players = tracks['players'][index]
            if not team_ready and len(players) >= 2:
                team_assigner.assign_team_color(frame, players)
                team_ready = True
            if team_ready:
                for player_id, track in players.items():
                    team_assigner.get_player_team(frame, track['bbox'], player_id)

        seen += len(chunk)
        print(f"  pass 1: {min(seen, n_expected)}/{n_expected} frames",
              end='\r', flush=True)
    print()

    if estimator is None:
        sys.exit("No frames could be read from that section.")
    n_frames = len(tracks['players'])
    if camera_movement is None:
        camera_movement = estimator.movement
        with open(camera_stub, 'wb') as handle:
            pickle.dump(camera_movement, handle)
    if not cached_tracks:
        with open(track_stub, 'wb') as handle:
            pickle.dump(tracks, handle)
    if not team_ready:
        sys.exit("No frame had two players in it; nothing to analyse.")

    # ---- From here to the second pass everything works on the tracks alone,
    # which stay a few megabytes however long the clip is.
    ball_candidates = tracks.pop('ball_candidates', None)
    tracker.add_position_to_tracks(tracks)
    estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    tracked = None
    if calibration is not None and follow:
        tracked = pitch_calibration.interpolate(calib_keys, n_frames,
                                                calibration.image_size)
        errors = [c.error for c in calib_keys.values() if c.error is not None]
        if errors:
            print(f"Tracked calibration: {np.mean(errors):.2f}px mean, "
                  f"{max(errors):.2f}px worst over {len(calib_keys)} refits")
        calibration = tracked[0]
        transformer = CalibratedTransformer(tracked)
    elif calibration is not None:
        transformer = CalibratedTransformer(calibration)
    else:
        transformer = ViewTransformer()
    transformer.add_transformed_position_to_tracks(tracks)

    # Pick the real ball out of the candidates. Most ball-like detections are
    # clutter beside the pitch - bottles and kit on the grass past the
    # touchline - and the calibration is what tells them apart.
    if ball_candidates and calibration is not None:
        selector = BallTracker(fps)
        chosen, raw = [], 0
        for frame_num, candidates in enumerate(ball_candidates):
            frame_cal = (tracked[min(frame_num, len(tracked) - 1)]
                         if tracked else calibration)
            entries = []
            for candidate in candidates.values():
                bbox = candidate['bbox']
                centre = np.array([[(bbox[0] + bbox[2]) / 2,
                                    (bbox[1] + bbox[3]) / 2]], np.float32)
                entries.append((bbox, candidate.get('conf', 0.0),
                                frame_cal.to_pitch(centre)[0]))
            raw += len(entries)
            pick = selector.update(entries)
            chosen.append({1: {'bbox': pick}} if pick is not None else {})
        kept = sum(1 for frame in chosen if frame)
        print(f"Ball: {raw} candidates -> kept in {kept}/{len(chosen)} frames "
              f"after discarding everything outside the lines")
        tracks['ball'] = chosen

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Teams were decided in the first pass while the frames were in hand;
    # writing them onto the tracks needs no frames at all.
    if not args.kit_colours:
        for team in TEAM_DISPLAY:
            kit = np.round(team_assigner.team_colors[team]).astype(int).tolist()
            name = {1: 'blue', 2: 'red'}[team]
            print(f"  team {team} drawn {name}; detected kit colour BGR {kit}")
    for player_track in tracks['players']:
        for player_id, track in player_track.items():
            team = team_assigner.player_team_dict.get(player_id, 1)
            track['team'] = team
            track['team_color'] = (team_assigner.team_colors[team]
                                   if args.kit_colours
                                   else TEAM_DISPLAY.get(team, (230, 230, 230)))

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
        assigned_player = -1
        if ball_bbox is not None and np.all(np.isfinite(ball_bbox)):
            assigned_player = player_assigner.assign_ball_to_player(player_track,
                                                                    ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1]
                                     if team_ball_control else 1)
    team_ball_control = np.array(team_ball_control)

    # ---- Second pass: draw each chunk and write it straight out, so the
    # annotated frames are never all in memory either.
    writer = open_writer(output_path, fps, info['width'], info['height'])
    renderer = PitchRenderer(scale=6) if calibration is not None else None
    seen = 0
    try:
        for chunk in iter_video(args.video, start, end, chunk=args.chunk):
            high = min(seen + len(chunk), n_frames)
            if high <= seen:
                break
            chunk = chunk[:high - seen]
            piece = {key: value[seen:high] for key, value in tracks.items()}

            frames = tracker.draw_annotations(chunk, piece, team_ball_control,
                                              in_place=True, offset=seen)
            frames = estimator.draw_camera_movement(
                frames, camera_movement[seen:high], in_place=True)
            speed_and_distance_estimator.draw_speed_and_distance(frames, piece)
            if calibration is not None and not args.no_pitch_view:
                add_pitch_view(frames, piece, calibration,
                               scale=args.pitch_view_scale,
                               team_shape=not args.no_team_shape,
                               tracked=tracked[seen:high] if tracked else None,
                               renderer=renderer)
            for frame in frames:
                writer.write(frame)
            seen = high
            print(f"  pass 2: {seen}/{n_frames} frames", end='\r', flush=True)
    finally:
        writer.release()
    print()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
