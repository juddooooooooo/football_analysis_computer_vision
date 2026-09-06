"""Click a few pitch landmarks to calibrate a clip.

Run it, click whichever of the offered landmarks you can see, and the fit is
refined against the painted lines and saved. Four is the minimum; six or
seven spread across the frame gives a noticeably better result than four
bunched in one corner.

    python -m pitch_calibration.interactive <video> --start 23:04 --end 23:35

Orientation convention: x = 0 is the goal towards the LEFT of the image,
y = 0 is the touchline towards the TOP. Pick landmarks consistently with
that and the fit will be right.
"""
import argparse
import sys

import cv2
import numpy as np

from utils import probe_video, parse_timestamp, format_timestamp, TimestampError
from . import pitch_model as pm
from .calibration import Calibration, refine, line_mask, save

_H = pm.WIDTH / 2

# name -> (x, y) in metres. Ordered roughly left to right across the pitch.
LANDMARKS = [
    ("left goal, top post base",            (0.0, _H - pm.GOAL_WIDTH / 2)),
    ("left goal, bottom post base",         (0.0, _H + pm.GOAL_WIDTH / 2)),
    ("left penalty area, top goal-line",    (0.0, 13.84)),
    ("left penalty area, bottom goal-line", (0.0, 54.16)),
    ("left penalty area, top outer corner", (16.5, 13.84)),
    ("left penalty area, bottom outer corner", (16.5, 54.16)),
    ("left 6-yard box, top outer corner",   (5.5, 24.84)),
    ("left 6-yard box, bottom outer corner", (5.5, 43.16)),
    ("left penalty spot",                   (11.0, _H)),
    ("halfway line meets top touchline",    (pm.LENGTH / 2, 0.0)),
    ("halfway line meets bottom touchline", (pm.LENGTH / 2, pm.WIDTH)),
    ("centre circle, leftmost point",       (pm.LENGTH / 2 - 9.15, _H)),
    ("centre circle, rightmost point",      (pm.LENGTH / 2 + 9.15, _H)),
    ("centre circle, topmost point",        (pm.LENGTH / 2, _H - 9.15)),
    ("centre circle, bottommost point",     (pm.LENGTH / 2, _H + 9.15)),
    ("centre spot",                         (pm.LENGTH / 2, _H)),
    ("right 6-yard box, top outer corner",  (99.5, 24.84)),
    ("right 6-yard box, bottom outer corner", (99.5, 43.16)),
    ("right penalty spot",                  (94.0, _H)),
    ("right penalty area, top outer corner", (88.5, 13.84)),
    ("right penalty area, bottom outer corner", (88.5, 54.16)),
    ("right penalty area, top goal-line",   (pm.LENGTH, 13.84)),
    ("right penalty area, bottom goal-line", (pm.LENGTH, 54.16)),
    ("right goal, top post base",           (pm.LENGTH, _H - pm.GOAL_WIDTH / 2)),
    ("right goal, bottom post base",        (pm.LENGTH, _H + pm.GOAL_WIDTH / 2)),
]


def grab_frame(video, at_second):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_second * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame at {format_timestamp(at_second)}")
    return frame


def collect(frame, show_lines=True):
    """Walk the landmark list; click what is visible, skip what is not."""
    base = frame.copy()
    if show_lines:
        m = line_mask(frame)
        base[m > 0] = (0, 240, 255)
    picked_img, picked_world = [], []
    state = {'i': 0, 'hover': None}

    def redraw():
        img = base.copy()
        for (x, y) in picked_img:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(x), int(y)), 9, (255, 255, 255), 1)
        name = LANDMARKS[state['i']][0] if state['i'] < len(LANDMARKS) else '-'
        bar = img.shape[0] - 46
        cv2.rectangle(img, (0, bar), (img.shape[1], img.shape[0]), (20, 20, 20), -1)
        cv2.putText(img, f"CLICK: {name}", (10, bar + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img,
                    f"[space]=skip  [u]=undo  [enter]=done ({len(picked_img)} picked, need 4)  [esc]=cancel",
                    (10, bar + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 180), 1)
        cv2.imshow('calibrate', img)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and state['i'] < len(LANDMARKS):
            picked_img.append((float(x), float(y)))
            picked_world.append(LANDMARKS[state['i']][1])
            state['i'] += 1
            redraw()

    cv2.namedWindow('calibrate', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('calibrate', min(1600, frame.shape[1]), min(950, frame.shape[0] + 60))
    cv2.setMouseCallback('calibrate', on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None, None
        if k == 32 and state['i'] < len(LANDMARKS):
            state['i'] += 1
            redraw()
        elif k in (ord('u'), 8) and picked_img:
            picked_img.pop()
            picked_world.pop()
            state['i'] = max(0, state['i'] - 1)
            redraw()
        elif k in (13, 10):
            if len(picked_img) >= 4:
                break
        if state['i'] >= len(LANDMARKS) and len(picked_img) >= 4:
            break
    cv2.destroyAllWindows()
    return np.float32(picked_img), np.float32(picked_world)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('--start', default=None, help='section start, e.g. 23:04')
    ap.add_argument('--end', default=None, help='section end, e.g. 23:35')
    ap.add_argument('--at', default=None,
                    help='frame to calibrate on (default: the section start)')
    ap.add_argument('--max-shift', type=float, default=30.0,
                    help='how far refinement may move a clicked point (px)')
    args = ap.parse_args()

    try:
        start = parse_timestamp(args.start) if args.start else None
        end = parse_timestamp(args.end) if args.end else None
        at = parse_timestamp(args.at) if args.at else (start or 0.0)
    except TimestampError as exc:
        sys.exit(str(exc))

    info = probe_video(args.video)
    window = None if start is None and end is None else (
        int(start or 0), int(end if end is not None else info['duration']))

    frame = grab_frame(args.video, at)
    img_pts, world_pts = collect(frame)
    if img_pts is None:
        sys.exit("Cancelled.")
    print(f"{len(img_pts)} landmarks clicked; refining against the painted lines...")

    H, err, n = refine(frame, img_pts, world_pts, max_shift=args.max_shift)
    cal = Calibration(H, [info['width'], info['height']], err)
    save(args.video, window, cal)
    print(f"Mean alignment error {err:.2f}px over {n} model points.")
    print(f"Saved for {args.video} {window if window else '(whole video)'}.")
    if err > 6:
        print("That is a loose fit. Re-run and click more landmarks, spread "
              "as widely across the frame as you can.")


if __name__ == '__main__':
    main()
