"""
metrics.py — Real-time ocular tracking metrics.

All coordinates are normalised [0, 1] (fraction of screen/canvas size).

Metrics computed
----------------
accuracy    : mean Euclidean error (gaze vs target), 0 = perfect, 1 = worst
smoothness  : 1 - normalised velocity variance  (1 = perfectly smooth)
latency_ms  : estimated gaze lag behind target via cross-correlation
stability   : measures tracking jitter (variance of the error)
score       : 0-100 composite score suitable for progress tracking
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple

_EPS = 1e-9

# Rolling window lengths
_WIN_ACCURACY   = 60   # frames (~1 s at 60 fps)
_WIN_SMOOTH     = 30
_WIN_STABILITY  = 30
_MAX_LAG_FRAMES = 45   # max lag to search


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
        ts = len(self._gaze_buf)

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

        # Stability: track the error variance to measure jitter
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
        smooth = 1.0 - np.clip(var / 0.05, 0.0, 1.0)
        return float(smooth)

    @property
    def smoothness_pct(self) -> float:
        return float(self.smoothness * 100.0)

    @property
    def latency_frames(self) -> int:
        """
        Estimated gaze lag (in frames) via Pearson cross-correlation.
        """
        if len(self._gaze_buf) < _MAX_LAG_FRAMES * 2:
            return 0
            
        g = np.array([p[0] for p in self._gaze_buf])
        t = np.array([p[0] for p in self._target_buf])
        
        # Avoid division by zero if there's no movement
        if np.std(g) < 1e-3 or np.std(t) < 1e-3:
            return 0
            
        best_lag = 0
        max_corr = -float('inf')
        
        # Shift target back in time to see where it best aligns with current gaze
        for lag in range(_MAX_LAG_FRAMES + 1):
            if lag == 0:
                corr = np.corrcoef(g, t)[0, 1]
            else:
                corr = np.corrcoef(g[lag:], t[:-lag])[0, 1]
                
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
                
        # If correlation is weak, don't trust the lag calculation
        if max_corr < 0.3:
            return 0
            
        return best_lag

    @property
    def latency_ms(self) -> float:
        """
        Approximate lag in milliseconds. 
        Frontend requestAnimationFrame emits at ~60fps (1000/60 ms per frame).
        """
        return float(self.latency_frames * (1000.0 / 60.0))

    @property
    def stability(self) -> float:
        """
        Standard deviation of the Euclidean error over the recent window.
        Measures how consistent the tracking is (jitter). Lower = more stable.
        """
        if len(self._stability_buf) < 3:
            return 0.0
        return float(np.std(self._stability_buf))

    @property
    def stability_pct(self) -> float:
        """Stability score 0–100 where 100 = rock-steady (low variance)."""
        # A standard deviation of 0.1 in screen fraction is high instability.
        return float(np.clip((1.0 - self.stability / 0.1) * 100.0, 0.0, 100.0))

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