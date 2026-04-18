"""
calibration.py — Maps raw gaze (iris ratios + head-pose compensated)
to screen space using a ridge-regularized polynomial fit.

Why ridge regression:
  With 9 calibration points and degree-3 features (10 coefficients),
  plain least-squares is underdetermined. Ridge (Tikhonov) adds a
  small L2 penalty, which makes the fit stable without biasing the
  low-order terms that dominate the mapping.

Why degree-3:
  Degree-2 has only ONE cross-term (xy) — on a 3x3 calibration grid
  this term is effectively fit from 4 samples, making diagonal
  mappings fragile. Degree-3 adds x²y and xy², giving the fit
  enough freedom to handle diagonal motion correctly.
"""

import numpy as np
from typing import List, Tuple, Optional


class CalibrationManager:

    def __init__(self, poly_degree: int = 2, ridge_lambda: float = 1e-4):
        """
        poly_degree   : 2 is usually sufficient. Bump to 3 only if you
                        have >12 calibration points with good coverage.
        ridge_lambda  : regularization strength. Small positive value
                        stabilizes the fit against noisy corner samples
                        without meaningfully biasing the mapping.
        """
        self.degree = poly_degree
        self.ridge_lambda = ridge_lambda
        self._samples: List[dict] = []
        self._coeff_x: Optional[np.ndarray] = None
        self._coeff_y: Optional[np.ndarray] = None
        self.is_fitted = False

    # ------------------------------------------------------------------ #

    def add_sample(self, screen_x: float, screen_y: float,
                   raw_x: float, raw_y: float):
        self._samples.append({
            "screen": (screen_x, screen_y),
            "raw":    (raw_x, raw_y),
        })

    def fit(self) -> bool:
        """
        Fit polynomial regression from raw gaze → screen coords using
        ridge regression (stable with few samples).

        Also upweights corner points during fitting because they
        carry the most information about the cross-terms that handle
        diagonal motion.
        """
        # Need at least as many samples as degree-2 basis (6) — degree 3
        # (10 basis) is underdetermined but ridge handles that.
        if len(self._samples) < 6:
            return False

        raw    = np.array([s["raw"]    for s in self._samples])
        screen = np.array([s["screen"] for s in self._samples])

        features = self._poly_features(raw)
        n_features = features.shape[1]

        # Weight corner samples 2x — they inform the cross-terms that
        # control diagonal mapping. Corners = samples where both
        # screen x and screen y are near 0 or 1.
        weights = np.ones(len(self._samples))
        for i, (sx, sy) in enumerate(screen):
            is_corner = (sx < 0.25 or sx > 0.75) and (sy < 0.25 or sy > 0.75)
            if is_corner:
                weights[i] = 2.0

        W = np.diag(weights)

        # Ridge regression: (X^T W X + λI) β = X^T W y
        # Don't regularize the bias term (column 0)
        reg = np.eye(n_features) * self.ridge_lambda
        reg[0, 0] = 0.0

        try:
            A = features.T @ W @ features + reg
            self._coeff_x = np.linalg.solve(A, features.T @ W @ screen[:, 0])
            self._coeff_y = np.linalg.solve(A, features.T @ W @ screen[:, 1])
            self.is_fitted = True
            return True
        except np.linalg.LinAlgError:
            return False

    def map(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        if not self.is_fitted:
            return float(raw_x), float(raw_y)
        feat = self._poly_features(np.array([[raw_x, raw_y]]))
        sx = float(np.clip((feat @ self._coeff_x).item(), 0.0, 1.0))
        sy = float(np.clip((feat @ self._coeff_y).item(), 0.0, 1.0))
        return sx, sy

    def reset(self):
        self._samples = []
        self._coeff_x = None
        self._coeff_y = None
        self.is_fitted = False

    def sample_count(self) -> int:
        return len(self._samples)

    def to_dict(self) -> dict:
        return {
            "fitted": self.is_fitted,
            "degree": self.degree,
            "n_samples": len(self._samples),
            "coeff_x": self._coeff_x.tolist() if self._coeff_x is not None else None,
            "coeff_y": self._coeff_y.tolist() if self._coeff_y is not None else None,
        }

    # ------------------------------------------------------------------ #

    def _poly_features(self, raw: np.ndarray) -> np.ndarray:
        """
        Build polynomial feature matrix up to self.degree for 2-D input.
        Degree 2: 1, x, y, x², xy, y²               (6 features)
        Degree 3: 1, x, y, x², xy, y², x³, x²y, xy², y³  (10 features)
        """
        x = raw[:, 0]
        y = raw[:, 1]
        cols = [np.ones(len(x))]
        for d in range(1, self.degree + 1):
            for k in range(d + 1):
                cols.append((x ** (d - k)) * (y ** k))
        return np.column_stack(cols)