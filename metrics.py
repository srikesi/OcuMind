"""
metrics.py — Real-time ocular tracking metrics.

All coordinates are normalised [0, 1] (fraction of screen/canvas size).

Metrics computed
----------------
accuracy    : mean Euclidean error (gaze vs target), 0 = perfect, 1 = worst
smoothness  : 1 - normalised velocity variance  (1 = perfectly smooth)
latency_ms  : estimated gaze lag behind target via cross-correlation
stability   : mean fixation drift during stationary target phases
score       : 0-100 composite score suitable for progress tracking
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple


_EPS = 1e-9

# Rolling window lengths
_WIN_ACCURACY   = 60   # frames (~2 s at 30 fps)
_WIN_SMOOTH     = 30
_WIN_STABILITY  = 30
_MAX_LAG_FRAMES = 45   # max lag to search (1.5 s)


class MetricsEngine:

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ #
    #  Data ingestion                                                      #
    # ------------------------------------------------------------------ #

    def update(self, gaze_x: float, gaze_y: float,
               target_x: float, target_y: float,
               target_moving: bool = True):
        """
        Call once per frame with calibrated gaze + current target position.
        """
        ts = len(self._gaze_buf)          # virtual frame index

        self._gaze_buf.append((gaze_x, gaze_y))
        self._target_buf.append((target_x, target_y))

        # Euclidean error this frame
        err = _dist(gaze_x, gaze_y, target_x, target_y)
        self._error_buf.append(err)

        # Gaze velocity
        if len(self._gaze_buf) >= 2:
            prev = self._gaze_buf[-2]
            vel  = _dist(gaze_x, gaze_y, prev[0], prev[1])
            self._vel_buf.append(vel)

        # Stability: accumulate only when target is stationary
        if not target_moving:
            self._stability_buf.append(err)

        self._frame_count += 1

    def reset(self):
        self._gaze_buf    = deque(maxlen=_MAX_LAG_FRAMES * 3)
        self._target_buf  = deque(maxlen=_MAX_LAG_FRAMES * 3)
        self._error_buf   = deque(maxlen=_WIN_ACCURACY)
        self._vel_buf     = deque(maxlen=_WIN_SMOOTH)
        self._stability_buf = deque(maxlen=_WIN_STABILITY)
        self._frame_count = 0
        self._session_errors = []

    # ------------------------------------------------------------------ #
    #  Live metric accessors                                               #
    # ------------------------------------------------------------------ #

    @property
    def accuracy(self) -> float:
        """Mean gaze error in last window, mapped to 0–1 (lower = better)."""
        if not self._error_buf:
            return 1.0
        return float(np.mean(self._error_buf))

    @property
    def accuracy_pct(self) -> float:
        """Accuracy expressed as 0–100 where 100 = perfect."""
        return float(np.clip((1.0 - self.accuracy / 0.5) * 100.0, 0.0, 100.0))

    @property
    def smoothness(self) -> float:
        """
        0–1 where 1 = perfectly smooth gaze path.
        Computed as 1 – normalised velocity standard deviation.
        """
        if len(self._vel_buf) < 3:
            return 1.0
        vels = np.array(self._vel_buf)
        var  = float(np.std(vels))
        # Normalise: typical jitter std ~0.03 → score 0.4, std ~0 → score 1.0
        smooth = 1.0 - np.clip(var / 0.05, 0.0, 1.0)
        return float(smooth)

    @property
    def smoothness_pct(self) -> float:
        return float(self.smoothness * 100.0)

    @property
    def latency_frames(self) -> int:
        """
        Estimated gaze lag (in frames) via cross-correlation of target and
        gaze x-trajectories. Returns 0 if insufficient data.
        """
        if len(self._gaze_buf) < _MAX_LAG_FRAMES * 2:
            return 0
        g = np.array([p[0] for p in self._gaze_buf])
        t = np.array([p[0] for p in self._target_buf])
        corr = np.correlate(g - g.mean(), t - t.mean(), mode="full")
        lags  = np.arange(-(len(g) - 1), len(t))
        # Only look at positive lags (gaze behind target)
        pos_mask = (lags >= 0) & (lags <= _MAX_LAG_FRAMES)
        if not pos_mask.any():
            return 0
        best = int(lags[pos_mask][np.argmax(corr[pos_mask])])
        return best

    @property
    def latency_ms(self) -> float:
        """Approximate lag in milliseconds (assumes 30 fps)."""
        return float(self.latency_frames * (1000.0 / 30.0))

    @property
    def stability(self) -> float:
        """
        Mean fixation error during stationary target phases.
        Lower = more stable fixation.
        """
        if not self._stability_buf:
            return 0.0
        return float(np.mean(self._stability_buf))

    @property
    def stability_pct(self) -> float:
        """Stability score 0–100 where 100 = rock-steady."""
        return float(np.clip((1.0 - self.stability / 0.3) * 100.0, 0.0, 100.0))

    @property
    def score(self) -> float:
        """
        Composite 0–100 score.
        Weighted average: accuracy 50%, smoothness 30%, stability 20%.
        """
        return float(
            0.50 * self.accuracy_pct +
            0.30 * self.smoothness_pct +
            0.20 * self.stability_pct
        )

    def snapshot(self) -> dict:
        """Return all metrics as a JSON-serialisable dict."""
        return {
            "accuracy":      round(self.accuracy_pct,    1),
            "smoothness":    round(self.smoothness_pct,  1),
            "latency_ms":    round(self.latency_ms,      1),
            "stability":     round(self.stability_pct,   1),
            "score":         round(self.score,            1),
            "frame":         self._frame_count,
        }


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _dist(x1, y1, x2, y2) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
