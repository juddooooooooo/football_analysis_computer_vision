"""Draw a bird's-eye view of the pitch with players on it.

The static pitch is rendered once and reused, so per-frame work is only the
markers. Positions come in as pitch metres, which is what the calibration
produces, so nothing here needs to know about the camera.
"""
import cv2
import numpy as np

from pitch_calibration import pitch_model as pm

GRASS = (58, 122, 62)
PAINT = (235, 235, 235)
BALL = (255, 255, 255)


class PitchRenderer:
    """Renders pitch-space positions onto a top-down pitch image."""

    def __init__(self, scale=5.0, margin=3.0):
        self.scale = scale                     # pixels per metre
        self.margin = margin                   # metres of surround
        self.width = int((pm.LENGTH + 2 * margin) * scale)
        self.height = int((pm.WIDTH + 2 * margin) * scale)
        self._base = self._draw_pitch()

    def _to_px(self, points):
        p = np.asarray(points, float).reshape(-1, 2)
        x = (p[:, 0] + self.margin) * self.scale
        y = (p[:, 1] + self.margin) * self.scale
        return np.stack([x, y], axis=1)

    def _draw_pitch(self):
        img = np.full((self.height, self.width, 3), GRASS, np.uint8)
        # Mowing stripes, purely so the view reads as a pitch at a glance.
        band = int(self.width / 14)
        for i in range(0, self.width, band * 2):
            img[:, i:i + band] = np.clip(
                img[:, i:i + band].astype(int) + 10, 0, 255).astype(np.uint8)
        for line in pm.polylines():
            pts = self._to_px(line).astype(np.int32)
            cv2.polylines(img, [pts], False, PAINT, 2, cv2.LINE_AA)
        for spot in self._to_px(pm.spots()).astype(int):
            cv2.circle(img, tuple(spot), 3, PAINT, -1, cv2.LINE_AA)
        return img

    def blank(self):
        return self._base.copy()

    def draw_players(self, img, positions, colour, radius=6, labels=None):
        """Plot one team. positions is (N,2) in metres, colour is BGR."""
        pts = self._to_px(positions).astype(int)
        colour = tuple(int(c) for c in colour)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), radius, colour, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), radius, (20, 20, 20), 1, cv2.LINE_AA)
            if labels is not None:
                cv2.putText(img, str(labels[i]), (x + radius + 1, y - radius),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (245, 245, 245), 1,
                            cv2.LINE_AA)
        return img

    def draw_ball(self, img, position, radius=4):
        x, y = self._to_px(position).astype(int)[0]
        cv2.circle(img, (x, y), radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, BALL, -1, cv2.LINE_AA)
        return img

    def draw_shape(self, img, positions, colour, hull=True, centroid=True):
        """Convex hull and centroid of a team: the shape of the block."""
        pts = np.asarray(positions, float).reshape(-1, 2)
        if len(pts) < 3:
            return img
        colour = tuple(int(c) for c in colour)
        if hull:
            px = self._to_px(pts).astype(np.int32)
            h = cv2.convexHull(px)
            overlay = img.copy()
            cv2.fillPoly(overlay, [h], colour)
            cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)
            cv2.polylines(img, [h], True, colour, 1, cv2.LINE_AA)
        if centroid:
            c = self._to_px(pts.mean(axis=0)).astype(int)[0]
            cv2.drawMarker(img, tuple(c), colour, cv2.MARKER_CROSS, 11, 2)
        return img


def inset(frame, panel, scale=0.34, margin=12, alpha=0.85):
    """Composite the pitch view into the bottom-left of a video frame."""
    fh, fw = frame.shape[:2]
    w = int(fw * scale)
    h = int(panel.shape[0] * w / panel.shape[1])
    small = cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)
    y1, y0 = fh - margin, fh - margin - h
    x0, x1 = margin, margin + w
    if y0 < 0 or x1 > fw:
        return frame
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (25, 25, 25), -1)
    roi = frame[y0:y1, x0:x1]
    cv2.addWeighted(small, alpha, roi, 1 - alpha, 0, roi)
    return frame
