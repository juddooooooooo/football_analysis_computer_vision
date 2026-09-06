"""Map image pixels to pitch metres, and remember the mapping per clip.

A calibration is a homography from image coordinates to pitch metres. It is
found from four landmark correspondences and then refined by aligning the
whole pitch model against the painted lines the camera can actually see, so
accuracy does not fall away in parts of the frame far from the landmarks.

Calibrations are stored in calibrations.json keyed by video and section,
because the camera reframes over the course of a match.
"""
import json
import os

import cv2
import numpy as np
from scipy.optimize import minimize

from . import pitch_model

STORE = 'calibrations.json'


def grass_mask(frame):
    """The playing surface, as one filled region.

    Everything outside it — crowd, boards, dugouts — has to be excluded
    before looking for lines, because that texture lights up a top-hat just
    as brightly as paint does.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    if n <= 1:
        return green
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pitch = (labels == biggest).astype(np.uint8) * 255
    # Close over the players standing on it, so their outlines are not holes.
    return cv2.morphologyEx(pitch, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))


def line_mask(frame, pitch=None):
    """Painted lines as a binary mask.

    Uses a top-hat, which keeps thin structures brighter than their local
    surroundings. Absolute white thresholding fails here: floodlit grass is
    brighter than the lines in shadowed parts of the frame.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    top = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    mask = (top > 15).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Drop compact blobs. Players in light kit light up a top-hat exactly as
    # paint does, and a shirt sitting on a line drags the fit towards it.
    # Paint is a few pixels wide and vanishes under this erosion; a player
    # does not, so whatever survives is not a line.
    thick = cv2.erode(mask, np.ones((5, 5), np.uint8))
    thick = cv2.dilate(thick, np.ones((9, 9), np.uint8))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(thick))

    if pitch is None:
        pitch = grass_mask(frame)
    # Erode the pitch a little: the boards meet the grass in a bright seam
    # that would otherwise read as a touchline.
    pitch = cv2.erode(pitch, np.ones((9, 9), np.uint8))
    return cv2.bitwise_and(mask, pitch)


def _homography(image_pts, world_pts):
    src = np.float32(world_pts).reshape(-1, 1, 2)
    dst = np.float32(image_pts).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, method=0)
    return H  # pitch -> image


CAP = 40.0      # px; distances are clipped here
DELTA = 6.0     # px; Huber knee


def _project(model, H, shape):
    h, w = shape
    pts = cv2.perspectiveTransform(
        model.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return pts, inside


def refine(frame, image_pts, world_pts, mask=None, spacing=0.5, max_shift=10.0):
    """Nudge the four image landmarks so the whole model lines up.

    The set of model points being scored is frozen at whatever the initial
    landmarks make visible. Without that the optimiser has a trivial escape:
    swing the pitch out of frame until barely anything is scored, which looks
    like a near-perfect fit and is nonsense. Landmarks are also held within
    max_shift pixels, since they come from real detected line intersections.

    Returns (pitch->image homography, mean pixel error, points scored).
    """
    if mask is None:
        mask = line_mask(frame)
    shape = mask.shape
    # Distance to the nearest detected line pixel, capped so that a model
    # line crossing an unpainted area cannot dominate the fit.
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    dist = np.minimum(dist, CAP)
    model = pitch_model.sample_points(spacing)
    start = np.float32(image_pts).ravel()

    n_anchor = len(np.asarray(image_pts).reshape(-1, 2))
    H0 = _homography(image_pts, world_pts)
    _, scored = _project(model, H0, shape)
    if scored.sum() < 200:
        raise ValueError("Initial landmarks put almost no pitch in view.")
    model = model[scored]          # frozen scoring set

    def loss_of(H):
        pts, inside = _project(model, H, shape)
        d = np.full(len(pts), CAP)  # leaving the frame costs the cap
        if inside.any():
            xi = pts[inside, 0].astype(int)
            yi = pts[inside, 1].astype(int)
            d[inside] = dist[yi, xi]
        return np.where(d <= DELTA, 0.5 * d ** 2,
                        DELTA * (d - 0.5 * DELTA)).mean()

    def score(flat):
        H = _homography(flat.reshape(n_anchor, 2), world_pts)
        if H is None or not np.isfinite(H).all():
            return 1e6
        return float(loss_of(H))

    bounds = [(v - max_shift, v + max_shift) for v in start]
    best = minimize(score, start, method='Powell', bounds=bounds,
                    options={'xtol': 0.05, 'ftol': 1e-4, 'maxiter': 20000})
    H = _homography(best.x.reshape(n_anchor, 2), world_pts)
    if score(best.x) > score(start):     # never return a worse fit
        H = H0

    pts, inside = _project(model, H, shape)
    err = float(dist[pts[inside, 1].astype(int),
                     pts[inside, 0].astype(int)].mean())
    return H, err, int(inside.sum())


class Calibration:
    """A pitch<->image mapping for one clip."""

    def __init__(self, H_pitch_to_image, image_size=None, error=None):
        self.H_pitch_to_image = np.asarray(H_pitch_to_image, float)
        self.H_image_to_pitch = np.linalg.inv(self.H_pitch_to_image)
        self.image_size = image_size
        self.error = error

    def to_pitch(self, points):
        """Image pixels -> pitch metres. Accepts (N,2), returns (N,2)."""
        pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H_image_to_pitch)
        return out.reshape(-1, 2)

    def to_image(self, points):
        """Pitch metres -> image pixels."""
        pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H_pitch_to_image)
        return out.reshape(-1, 2)

    def on_pitch(self, points, margin=5.0):
        """Which points fall inside the pitch (plus a tolerance)."""
        p = np.asarray(points, float).reshape(-1, 2)
        return ((p[:, 0] > -margin) & (p[:, 0] < pitch_model.LENGTH + margin) &
                (p[:, 1] > -margin) & (p[:, 1] < pitch_model.WIDTH + margin))

    def as_dict(self):
        return {'H': self.H_pitch_to_image.tolist(),
                'image_size': self.image_size,
                'error': self.error}

    @classmethod
    def from_dict(cls, d):
        return cls(d['H'], d.get('image_size'), d.get('error'))


class CalibratedTransformer:
    """Drop-in replacement for ViewTransformer backed by a calibration.

    ViewTransformer carries four pixel coordinates hardcoded to one 1920x1080
    clip, so on any other footage it maps players to nonsense or drops them
    for falling outside its polygon. This uses the fitted homography instead
    and works on whatever clip it was calibrated for.
    """

    def __init__(self, calibration):
        # Either one Calibration for the clip, or one per frame from track().
        self.per_frame = isinstance(calibration, (list, tuple))
        self.calibration = calibration

    def for_frame(self, frame_num):
        if not self.per_frame:
            return self.calibration
        return self.calibration[min(frame_num, len(self.calibration) - 1)]

    def add_transformed_position_to_tracks(self, tracks):
        for obj, object_tracks in tracks.items():
            if obj == 'ball_candidates':
                continue
            for frame_num, track in enumerate(object_tracks):
                if not track:
                    continue
                cal = self.for_frame(frame_num)
                ids = list(track.keys())
                # A tracked calibration already follows the camera, so the
                # raw position is what it expects; only a fixed calibration
                # needs the movement subtracted first.
                field = 'position' if self.per_frame else 'position_adjusted'
                pts = np.array([track[i][field] for i in ids],
                               dtype=np.float32)
                pitch = cal.to_pitch(pts)
                ok = cal.on_pitch(pitch)
                for i, tid in enumerate(ids):
                    tracks[obj][frame_num][tid]['position_transformed'] = (
                        pitch[i].tolist() if ok[i] else None)


# Candidate world points for re-anchoring a tracked fit. Any four define a
# homography, so these need not be the points originally clicked - only ones
# that land inside the frame and spread out well across it.
_CANDIDATES = np.array([
    [88.5, 13.84], [88.5, 54.16], [105.0, 13.84], [105.0, 54.16],
    [99.5, 24.84], [99.5, 43.16], [94.0, 34.0],
    [16.5, 13.84], [16.5, 54.16], [0.0, 13.84], [0.0, 54.16],
    [5.5, 24.84], [5.5, 43.16], [11.0, 34.0],
    [52.5, 0.0], [52.5, 68.0], [52.5, 34.0],
    [43.35, 34.0], [61.65, 34.0], [52.5, 24.85], [52.5, 43.15],
    [0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0],
], float)


def _anchors_for(H, shape, want=6):
    """Pick spread-out world points that project inside the frame."""
    h, w = shape
    pts = cv2.perspectiveTransform(
        _CANDIDATES.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    ok = ((pts[:, 0] > 0) & (pts[:, 0] < w) & (pts[:, 1] > 0) & (pts[:, 1] < h))
    idx = np.flatnonzero(ok)
    if len(idx) < 4:
        return None, None
    # Greedy spread: keep taking the candidate furthest from those chosen.
    chosen = [int(idx[np.argmin(pts[idx, 0])])]
    while len(chosen) < min(want, len(idx)):
        rest = [i for i in idx if i not in chosen]
        d = [min(np.linalg.norm(pts[i] - pts[c]) for c in chosen) for i in rest]
        chosen.append(int(rest[int(np.argmax(d))]))
    return pts[chosen], _CANDIDATES[chosen]


def refine_from(frame, calibration, max_shift=35.0):
    """Re-fit onto this frame, seeded from an existing calibration.

    Anchors are derived from the seed rather than clicked again, so this can
    follow a camera through a clip without any manual input.
    """
    img_a, wld_a = _anchors_for(calibration.H_pitch_to_image, frame.shape[:2])
    if img_a is None:
        return calibration
    try:
        H, err, _ = refine(frame, img_a, wld_a, max_shift=max_shift)
    except ValueError:
        return calibration
    return Calibration(H, calibration.image_size, err)


def interpolate(keys, n_frames, image_size=None):
    """Expand {frame index: Calibration} to one Calibration per frame."""
    if not keys:
        raise ValueError("No keyframe calibrations to interpolate from.")
    marks = sorted(keys)
    out = []
    for i in range(n_frames):
        hi = np.searchsorted(marks, i, side='left')
        if hi == 0 or marks[min(hi, len(marks) - 1)] == i:
            mark = marks[min(hi, len(marks) - 1)]
            H_i, err = keys[mark].H_pitch_to_image, keys[mark].error
        else:
            a, b = marks[hi - 1], marks[min(hi, len(marks) - 1)]
            t = 0.0 if b == a else (i - a) / (b - a)
            H_i = ((1 - t) * keys[a].H_pitch_to_image
                   + t * keys[b].H_pitch_to_image)
            H_i = H_i / H_i[2, 2]
            err = keys[b].error
        out.append(Calibration(H_i, image_size, err))
    return out


def track(frames, calibration, every=10, max_shift=35.0, report=None):
    """A calibration per frame, for a camera that pans during the clip.

    A single fit decays as the camera moves: a pan changes perspective, so a
    translation offset cannot undo it. This re-fits every `every` frames,
    seeded from the previous fit, and interpolates in between.

    Returns a list of Calibration, one per frame.
    """
    shape = frames[0].shape[:2]
    H = calibration.H_pitch_to_image
    keys, errs = {}, {}
    for i in range(0, len(frames), every):
        img_a, wld_a = _anchors_for(H, shape)
        if img_a is None:
            keys[i] = H
            continue
        try:
            H_new, err, _ = refine(frames[i], img_a, wld_a, max_shift=max_shift)
        except ValueError:
            H_new, err = H, None
        keys[i] = H_new
        errs[i] = err
        H = H_new
        if report:
            report(i, err)
    if len(frames) - 1 not in keys:
        keys[len(frames) - 1] = H

    marks = sorted(keys)
    out = []
    for i in range(len(frames)):
        hi = np.searchsorted(marks, i, side='left')
        if hi == 0 or marks[min(hi, len(marks) - 1)] == i:
            H_i = keys[marks[min(hi, len(marks) - 1)]]
        else:
            a, b = marks[hi - 1], marks[min(hi, len(marks) - 1)]
            t = 0.0 if b == a else (i - a) / (b - a)
            H_i = (1 - t) * keys[a] + t * keys[b]
            H_i = H_i / H_i[2, 2]
        out.append(Calibration(H_i, calibration.image_size,
                               errs.get(marks[min(hi, len(marks) - 1)])))
    return out


def key_for(video, window):
    stem = os.path.splitext(os.path.basename(video))[0]
    return stem if window is None else f"{stem}_{int(window[0])}-{int(window[1])}"


def load(video, window, store=STORE):
    """The calibration for this clip, or None.

    Falls back to a calibration stored for the whole video when the exact
    section has none of its own.
    """
    if not os.path.exists(store):
        return None
    with open(store) as f:
        data = json.load(f)
    for k in (key_for(video, window), key_for(video, None)):
        if k in data:
            return Calibration.from_dict(data[k])
    return None


def save(video, window, calibration, store=STORE):
    data = {}
    if os.path.exists(store):
        with open(store) as f:
            data = json.load(f)
    data[key_for(video, window)] = calibration.as_dict()
    with open(store, 'w') as f:
        json.dump(data, f, indent=2)
