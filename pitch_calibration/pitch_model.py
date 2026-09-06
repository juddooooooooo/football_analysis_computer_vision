"""Geometry of a regulation pitch, in metres.

Origin is the corner flag; x runs goal line to goal line (0..LENGTH), y runs
touchline to touchline (0..WIDTH). Every marking here is a Laws of the Game
dimension, so the same model fits any regulation ground.
"""
import numpy as np

LENGTH = 105.0
WIDTH = 68.0

CENTRE_RADIUS = 9.15
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DIST = 11.0
GOAL_WIDTH = 7.32
CORNER_ARC_RADIUS = 1.0

_HALF = WIDTH / 2
_PA_TOP = _HALF - PENALTY_AREA_WIDTH / 2      # 13.84
_PA_BOT = _HALF + PENALTY_AREA_WIDTH / 2      # 54.16
_GA_TOP = _HALF - GOAL_AREA_WIDTH / 2         # 24.84
_GA_BOT = _HALF + GOAL_AREA_WIDTH / 2         # 43.16


def _arc(cx, cy, r, a0, a1, n=64):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def polylines():
    """Every painted marking, as a list of (N,2) point arrays in metres."""
    L, W = LENGTH, WIDTH
    out = [
        np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),   # touch/goal lines
        np.array([[L / 2, 0], [L / 2, W]], float),                   # halfway
        _arc(L / 2, _HALF, CENTRE_RADIUS, 0, 2 * np.pi),             # centre circle
    ]
    for side in (0, 1):
        gx = 0.0 if side == 0 else L                                 # goal line x
        d = 1 if side == 0 else -1                                   # into the pitch
        out.append(np.array([
            [gx, _PA_TOP], [gx + d * PENALTY_AREA_DEPTH, _PA_TOP],
            [gx + d * PENALTY_AREA_DEPTH, _PA_BOT], [gx, _PA_BOT]], float))
        out.append(np.array([
            [gx, _GA_TOP], [gx + d * GOAL_AREA_DEPTH, _GA_TOP],
            [gx + d * GOAL_AREA_DEPTH, _GA_BOT], [gx, _GA_BOT]], float))
        # penalty arc: only the part standing outside the penalty area
        spot_x = gx + d * PENALTY_SPOT_DIST
        edge = gx + d * PENALTY_AREA_DEPTH
        half = np.degrees(np.arccos(abs(edge - spot_x) / CENTRE_RADIUS))
        base = 0.0 if side == 0 else np.pi
        span = np.radians(half)
        out.append(_arc(spot_x, _HALF, CENTRE_RADIUS, base - span, base + span, 32))
    return out


def spots():
    """Centre spot and both penalty spots."""
    return np.array([[LENGTH / 2, _HALF],
                     [PENALTY_SPOT_DIST, _HALF],
                     [LENGTH - PENALTY_SPOT_DIST, _HALF]], float)


def goal_mouths():
    """The two goal lines between the posts, for drawing."""
    return [np.array([[0, _HALF - GOAL_WIDTH / 2], [0, _HALF + GOAL_WIDTH / 2]], float),
            np.array([[LENGTH, _HALF - GOAL_WIDTH / 2],
                      [LENGTH, _HALF + GOAL_WIDTH / 2]], float)]


def sample_points(spacing=0.5):
    """Dense points along every marking, for fitting against detected lines."""
    chunks = []
    for line in polylines():
        for a, b in zip(line[:-1], line[1:]):
            d = float(np.hypot(*(b - a)))
            n = max(int(d / spacing), 2)
            t = np.linspace(0, 1, n)[:, None]
            chunks.append(a + t * (b - a))
    return np.concatenate(chunks, axis=0)
