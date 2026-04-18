"""
calibration.py — Maps raw eye-relative gaze coordinates to screen space.

Uses a polynomial regression (degree-2) fitted to a grid of calibration
points where the user fixated at known screen positions.
"""

import numpy as np
from typing import List, Tuple, Optional


class CalibrationManager:
    """
    Collects fixation samples at known screen positions, then fits a
    polynomial warp that maps raw gaze (iris ratios) → screen (0–1).
    """

    def __init__(self, poly_degree: int = 2):
        self.degree = poly_degree
        self._samples: List[dict] = []   # {"screen": (sx,sy), "raw": (rx,ry)}
        self._coeff_x: Optional[np.ndarray] = None
        self._coeff_y: Optional[np.ndarray] = None
        self.is_fitted = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def add_sample(self, screen_x: float, screen_y: float,
                   raw_x: float, raw_y: float):
        """Record one calibration fixation."""
        self._samples.append({
            "screen": (screen_x, screen_y),
            "raw":    (raw_x, raw_y),
        })

    def fit(self) -> bool:
        """
        Fit polynomial regression from raw gaze → screen coords.
        Returns True if successful.
        Requires at least (degree+1)^2 samples.
        """
        if len(self._samples) < 4:
            return False

        raw    = np.array([s["raw"]    for s in self._samples])
        screen = np.array([s["screen"] for s in self._samples])

        features = self._poly_features(raw)

        # Least-squares fit independently for x and y
        try:
            self._coeff_x, _, _, _ = np.linalg.lstsq(features, screen[:, 0], rcond=None)
            self._coeff_y, _, _, _ = np.linalg.lstsq(features, screen[:, 1], rcond=None)
            self.is_fitted = True
            return True
        except np.linalg.LinAlgError:
            return False

    def map(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """
        Transform raw gaze → calibrated screen position.
        Falls back to a simple linear identity if not fitted.
        """
        if not self.is_fitted:
            return float(raw_x), float(raw_y)

        feat = self._poly_features(np.array([[raw_x, raw_y]]))
        sx = float(np.clip(feat @ self._coeff_x, 0.0, 1.0))
        sy = float(np.clip(feat @ self._coeff_y, 0.0, 1.0))
        return sx, sy

    def reset(self):
        self._samples = []
        self._coeff_x = None
        self._coeff_y = None
        self.is_fitted = False

    def sample_count(self) -> int:
        return len(self._samples)

    def to_dict(self) -> dict:
        """Serialise calibration state (for session export)."""
        return {
            "fitted": self.is_fitted,
            "n_samples": len(self._samples),
            "coeff_x": self._coeff_x.tolist() if self._coeff_x is not None else None,
            "coeff_y": self._coeff_y.tolist() if self._coeff_y is not None else None,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _poly_features(self, raw: np.ndarray) -> np.ndarray:
        """
        Build polynomial feature matrix up to self.degree for 2-D input.
        Columns: 1, x, y, x^2, xy, y^2  (for degree=2)
        """
        x = raw[:, 0]
        y = raw[:, 1]
        cols = [np.ones(len(x))]
        for d in range(1, self.degree + 1):
            for k in range(d + 1):
                cols.append((x ** (d - k)) * (y ** k))
        return np.column_stack(cols)
