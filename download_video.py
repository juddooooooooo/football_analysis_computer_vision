"""Download a YouTube clip as silent video for the analysis pipeline.

Grabs a video-only stream, so the file has no audio track. H.264/mp4 is
preferred because that is what OpenCV's VideoCapture reads most reliably.

A section can be selected either explicitly or at random:

    python download_video.py <url>
    python download_video.py <url> --start 12:30 --end 14:30
    python download_video.py <url> --random-clip
    python download_video.py <url> --random-clip --clip-duration 60 --seed 7

Sections are cut with ffmpeg when it is installed, so only the requested
part is fetched. Without ffmpeg the full stream is downloaded and trimmed
with OpenCV instead, which is slower and re-encodes the clip.
"""
import argparse
import os
import random
import sys

import cv2
import yt_dlp
from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor
from yt_dlp.utils import download_range_func

from utils import parse_timestamp, format_timestamp, resolve_window, TimestampError


def timestamp_arg(text):
    try:
        return parse_timestamp(text)
    except TimestampError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def build_format(max_height):
    """Video-only selectors, best codec for OpenCV first."""
    limit = f"[height<={max_height}]" if max_height else ""
    return "/".join([
        f"bestvideo{limit}[ext=mp4][vcodec^=avc1]",
        f"bestvideo{limit}[ext=mp4]",
        f"bestvideo{limit}",
    ])


def probe_duration(url):
    """Length in seconds, or None when the extractor does not report one."""
    with yt_dlp.YoutubeDL({"noplaylist": True, "quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("duration")


def download(url, outdir, name, max_height, window, use_ffmpeg):
    os.makedirs(outdir, exist_ok=True)
    opts = {
        "format": build_format(max_height),
        "outtmpl": os.path.join(outdir, name if name else "%(title).60s.%(ext)s"),
        "noplaylist": True,
        # No postprocessors: a single video-only stream needs no merge.
        "postprocessors": [],
    }

    # ffmpeg can fetch just the requested section. Without it, pull the whole
    # stream here and let trim_clip() cut it afterwards.
    if window and use_ffmpeg:
        opts["download_ranges"] = download_range_func(None, [window])
        opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def trim_clip(src, dst, start, end):
    """Cut [start, end) out of src with OpenCV. Re-encodes the video."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = "XVID" if dst.lower().endswith(".avi") else "mp4v"
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*codec),
                             fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    wanted = int((end - start) * fps)
    written = 0
    while written < wanted:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return written or None


def describe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    stats = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download a silent YouTube clip for the analysis pipeline.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--outdir", default="input_videos")
    parser.add_argument("--name", default=None,
                        help="output filename, e.g. match.mp4")
    parser.add_argument("--max-height", type=int, default=720,
                        help="cap resolution to keep inference fast (0 = no cap)")
    parser.add_argument("--start", type=timestamp_arg, default=None,
                        help="clip start, e.g. 90, 1:30 or 00:01:30")
    parser.add_argument("--end", type=timestamp_arg, default=None,
                        help="clip end; defaults to start + --clip-duration")
    parser.add_argument("--random-clip", action="store_true",
                        help="pick a random section instead of naming times")
    parser.add_argument("--clip-duration", type=float, default=120.0,
                        help="clip length in seconds (default: 120)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for --random-clip, to repeat a selection")
    parser.add_argument("--keep-full", action="store_true",
                        help="keep the full download when trimming without ffmpeg")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    duration = None
    if args.random_clip or args.start is not None or args.end is not None:
        duration = probe_duration(args.url)

    try:
        window = resolve_window(duration, args.start, args.end,
                                args.random_clip, args.clip_duration)
    except TimestampError as exc:
        sys.exit(str(exc))
    use_ffmpeg = FFmpegPostProcessor().available

    if window:
        start, end = window
        print(f"Section: {format_timestamp(start)} to {format_timestamp(end)} "
              f"({end - start:.0f}s)")
        if not use_ffmpeg:
            print("ffmpeg not found: downloading the full stream, then "
                  "trimming with OpenCV.")

    path = download(args.url, args.outdir, args.name, args.max_height,
                    window, use_ffmpeg)

    if not os.path.exists(path):
        sys.exit(f"Downloaded, but the expected file is missing: {path}")

    # Without ffmpeg the file above is the whole video, so cut it down here.
    if window and not use_ffmpeg:
        stem, ext = os.path.splitext(path)
        start, end = window
        clip_path = f"{stem}_clip_{int(start)}-{int(end)}{ext}"
        if trim_clip(path, clip_path, start, end) is None:
            sys.exit(f"Could not trim {path}. The full file is still there.")
        if args.keep_full:
            print(f"Kept full download: {path}")
        else:
            os.remove(path)
        path = clip_path

    stats = describe(path)
    if stats is None:
        sys.exit(f"Saved {path}, but OpenCV could not open it. "
                 f"Try --max-height 720 to force a different stream.")

    print(f"\nSaved: {path}")
    print("  {width}x{height}, {frames} frames @ {fps:.2f} fps".format(**stats))
    print(f"\nNext: python main.py \"{path}\" --random-clip")


if __name__ == "__main__":
    main()
