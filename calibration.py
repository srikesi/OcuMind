"""
calibration.py — Maps raw gaze coordinates to screen space using linear regression.
"""

import numpy as np
import threading

class CalibrationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._samples = []
            self._coeff_x = None
            self._coeff_y = None
            self.is_calibrated = False

    def add_sample(self, screen_x, screen_y, raw_x, raw_y):
        with self._lock:
            self._samples.append((screen_x, screen_y, raw_x, raw_y))

    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    def fit(self) -> bool:
        with self._lock:
            if len(self._samples) < 4:
                return False

            Y_x = np.array([s[0] for s in self._samples])
            Y_y = np.array([s[1] for s in self._samples])
            raw_x = np.array([s[2] for s in self._samples])
            raw_y = np.array([s[3] for s in self._samples])

            # MATH FIX: Changed to Bilinear features (removed x**2 and y**2)
            # This makes the calibration much more stable near the edges of the screen
            X = np.column_stack([
                np.ones_like(raw_x),
                raw_x,
                raw_y,
                raw_x * raw_y
            ])

            try:
                self._coeff_x, _, _, _ = np.linalg.lstsq(X, Y_x, rcond=None)
                self._coeff_y, _, _, _ = np.linalg.lstsq(X, Y_y, rcond=None)
                self.is_calibrated = True
                return True
            except np.linalg.LinAlgError:
                self.is_calibrated = False
                return False

    def map(self, raw_x, raw_y):
        with self._lock:
            if not self.is_calibrated:
                return raw_x, raw_y

            # Predict screen coordinates using the stable bilinear features
            feat = np.array([1.0, raw_x, raw_y, raw_x * raw_y])
            pred_x = feat @ self._coeff_x
            pred_y = feat @ self._coeff_y

            sx = float(np.clip(pred_x.item() if isinstance(pred_x, np.ndarray) else pred_x, 0.0, 1.0))
            sy = float(np.clip(pred_y.item() if isinstance(pred_y, np.ndarray) else pred_y, 0.0, 1.0))

            return sx, sy