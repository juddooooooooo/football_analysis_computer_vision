"""Helpers for selecting a section of a video by time."""
import random

MARGIN_FRACTION = 0.05  # skip intros/build-up when picking at random


class TimestampError(ValueError):
    """Raised when a timestamp or a clip window cannot be used."""


def parse_timestamp(text):
    """Accept SS, MM:SS or HH:MM:SS, with fractional seconds allowed."""
    parts = str(text).strip().split(":")
    if len(parts) > 3:
        raise TimestampError(f"not a timestamp: {text}")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise TimestampError(f"not a timestamp: {text}")
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    if seconds < 0:
        raise TimestampError(f"negative timestamp: {text}")
    return seconds


def format_timestamp(seconds):
    total = int(seconds)
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def choose_random_window(duration, clip_length):
    """A clip_length window placed randomly, away from the very ends."""
    if duration <= clip_length:
        return 0.0, duration
    low = duration * MARGIN_FRACTION
    high = duration * (1 - MARGIN_FRACTION) - clip_length
    if high <= low:  # short video: the margins would leave no room
        low, high = 0.0, duration - clip_length
    start = random.uniform(low, high)
    return start, start + clip_length


def resolve_window(duration, start=None, end=None, random_clip=False,
                   clip_duration=30.0):
    """Work out which section to use. Returns (start, end) or None for all.

    duration may be None when it is not known ahead of time; a random clip
    cannot be picked in that case.
    """
    if random_clip:
        if start is not None or end is not None:
            raise TimestampError(
                "--random-clip cannot be combined with --start/--end.")
        if duration is None:
            raise TimestampError(
                "Cannot pick a random clip: the video duration is unknown.")
        return choose_random_window(duration, clip_duration)

    if start is None and end is None:
        return None

    window_start = start or 0.0
    window_end = end if end is not None else window_start + clip_duration

    if window_end <= window_start:
        raise TimestampError(
            f"end ({format_timestamp(window_end)}) must come after "
            f"start ({format_timestamp(window_start)}).")
    if duration is not None:
        if window_start >= duration:
            raise TimestampError(
                f"start ({format_timestamp(window_start)}) is past the end "
                f"of the video ({format_timestamp(duration)}).")
        window_end = min(window_end, duration)
    return window_start, window_end
