import cv2

DEFAULT_FPS = 24.0


def probe_video(video_path):
    """Frame rate, size and length of a video without decoding all of it."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": frame_count / fps if fps else 0.0,
    }
    cap.release()
    return info


def read_video(video_path, start=None, end=None):
    """Read frames into memory, optionally only the [start, end) seconds.

    Reading a whole match at once needs far more memory than most machines
    have, so pass a window when working with long footage.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    wanted = None
    if start is not None or end is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        start = start or 0.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        if end is not None:
            wanted = int((end - start) * fps)

    frames = []
    while wanted is None or len(frames) < wanted:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(ouput_video_frames, output_video_path, fps=DEFAULT_FPS):
    if not ouput_video_frames:
        raise ValueError("No frames to save.")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
