"""Pick the real ball out of the candidate detections.

The detector finds several ball-like objects per frame. On the City clip,
72% of them sit outside the lines: water bottles and kit on the grass just
past the far touchline, at four positions that recur frame after frame.
Confidence does not separate them, some reach 0.76. Three things do:

  where it is    the ball is between the lines, the clutter is not
  how it moves   a ball cannot cross the pitch between two frames

Measured over 180 frames: the line filter alone removes 47 of 152 picks
and takes physically impossible jumps from 69 to 1; the motion gate takes
that last one to 0. A third test, rejecting anything that sits still, was
tried and dropped - it cost 7 real detections and caught nothing the line
filter had not, because a ball waiting at a set piece also sits still.

Nothing here needs the detector to improve; it is geometry and motion on
top of detections we already have.
"""
import numpy as np

from pitch_calibration import pitch_model as pm

MAX_SPEED = 38.0        # m/s. A struck ball reaches about 35.


class BallTracker:
    """Choose one ball per frame, and say where it will be next.

    Keeps a constant-velocity estimate, which both scores candidates and
    fills short gaps. It deliberately does not extrapolate far: a straight
    line is only true of a ball for a fraction of a second.
    """

    def __init__(self, fps, margin=0.4, max_coast=8):
        self.fps = float(fps)
        self.margin = margin          # metres of tolerance outside the lines
        self.max_coast = max_coast    # frames to carry on with no detection
        self.position = None          # last accepted position, in metres
        self.velocity = np.zeros(2)   # m/s
        self.missing = 0

    def _inside(self, point):
        m = self.margin
        return (-m <= point[0] <= pm.LENGTH + m and
                -m <= point[1] <= pm.WIDTH + m)

    def predict(self):
        """Where the ball should be now, from the last estimate."""
        if self.position is None:
            return None
        return self.position + self.velocity * (self.missing + 1) / self.fps

    def update(self, candidates):
        """candidates: [(bbox, confidence, pitch_xy)]. Returns the pick or None.

        Every candidate is scored; the ball is whichever best agrees with
        confidence, the motion estimate, and not being a bottle.
        """
        usable = []
        for bbox, conf, point in candidates:
            point = np.asarray(point, float)
            if self._inside(point):
                usable.append((bbox, conf, point))

        if not usable:
            self.missing += 1
            if self.position is not None and self.missing > self.max_coast:
                self.position = None          # too long, stop pretending
                self.velocity = np.zeros(2)
            return None

        expected = self.predict()
        best, best_score = None, -1e9
        for bbox, conf, point in usable:
            score = float(conf)
            if expected is not None:
                gap = float(np.linalg.norm(point - expected))
                reach = MAX_SPEED * (self.missing + 1) / self.fps
                if gap > reach:
                    continue                   # no ball travels that far
                score += 2.0 * (1.0 - gap / max(reach, 1e-6))
            if score > best_score:
                best, best_score = (bbox, point), score

        if best is None:                       # all of them failed the gate
            self.missing += 1
            return None

        bbox, point = best
        if self.position is not None:
            elapsed = (self.missing + 1) / self.fps
            measured = (point - self.position) / elapsed
            # Smooth, or a single noisy frame throws the estimate badly off.
            self.velocity = 0.5 * self.velocity + 0.5 * measured
        self.position = point
        self.missing = 0
        return bbox

    @property
    def coasting(self):
        """True while filling a gap from the motion estimate alone."""
        return self.position is not None and 0 < self.missing <= self.max_coast
