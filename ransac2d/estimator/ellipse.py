from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg
from scipy.optimize import least_squares

from ..utils._numba_utils import transform_to_ellipse_frame
from ..model import Ellipse
from .base import BaseEstimator


class EllipseEstimator(BaseEstimator):
    """
    Ellipse shape estimator: one class for unknown size (5-point) and known size (3-point).

    If size is None, uses 5-point (unknown size) method;
    if size=(a, b), uses 3-point known-size method.
    """

    def __init__(
        self,
        size: Optional[tuple[float, float]] = None,
        *,
        max_residual_sample: Optional[float] = None,
    ):
        """Initialize the estimator.

        Args:
            size: If None, fit full ellipse from 5 points. If (a, b) semi-axes,
                fit center and angle from 3 points.
            max_residual_sample: Max residual on minimal sample for known-size;
                if None, 1% of min(a, b).
        """
        self.size = size
        self.max_residual_sample = max_residual_sample
        if size is not None:
            self._a, self._b = float(size[0]), float(size[1])

    @property
    def min_samples(self) -> int:
        """Minimum number of points (3 for known size, 5 for full)."""
        return 3 if self.size is not None else 5

    def is_data_valid(self, sample: np.ndarray) -> bool:
        """Reject degenerate or collinear minimal samples."""
        n = 3 if self.size is not None else 5
        if sample.shape[0] != n:
            return False
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        return abs(v1[0] * v2[1] - v1[1] * v2[0]) >= (1e-6 if n == 3 else 1e-10)

    @staticmethod
    def _conic_to_parametric(
        A: float, B: float, C: float, D: float, E: float, F: float
    ) -> tuple[float, float, float, float, float]:
        """General ellipse equation -> (a, b, x0, y0, theta)."""
        divider = B**2 - 4 * A * C
        if abs(divider) < 1e-10:
            raise ValueError("Degenerate conic")
        disc = (A - C) ** 2 + B**2
        term = A * E**2 + C * D**2 - B * D * E + (B**2 - 4 * A * C) * F
        sqrt_a = 2 * term * ((A + C) + np.sqrt(disc))
        sqrt_b = 2 * term * ((A + C) - np.sqrt(disc))
        if sqrt_a < 0 or sqrt_b < 0:
            raise ValueError("Not an ellipse")
        a = -np.sqrt(sqrt_a) / divider
        b = -np.sqrt(sqrt_b) / divider
        x0 = (2 * C * D - B * E) / divider
        y0 = (2 * A * E - B * D) / divider
        theta = 0.5 * np.arctan2(-B, C - A)
        return float(a), float(b), float(x0), float(y0), float(theta)

    @staticmethod
    def _fit_from_5_points(pts: np.ndarray) -> tuple[float, float, float, float, float]:
        """Solve for (a, b, x0, y0, theta) from 5 points. F = 1 normalization."""
        if pts.shape[0] < 5:
            raise ValueError("Need at least 5 points")
        x = pts[:5, 0]
        y = pts[:5, 1]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-10:
            raise ValueError("Points are collinear or degenerate")
        X = np.zeros((5, 5), dtype=np.float64)
        X[:, 0] = x**2
        X[:, 1] = x * y
        X[:, 2] = y**2
        X[:, 3] = x
        X[:, 4] = y
        coeffs = linalg.solve(X, -np.ones(5, dtype=np.float64), check_finite=False)
        A, B, C, D, E = coeffs
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("Solution contains NaN or Inf")
        return EllipseEstimator._conic_to_parametric(A, B, C, D, E, 1.0)

    @staticmethod
    def _initial_guess_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, float]:
        """Rough center and angle from three points."""
        pts = np.vstack([p1, p2, p3])
        center = np.mean(pts, axis=0)
        pts_c = pts - center
        try:
            cov = np.cov(pts_c.T)
            vals, vecs = np.linalg.eig(cov)
            main_dir = np.real(vecs[:, int(np.argmax(np.real(vals)))])
            theta = float(np.arctan2(main_dir[1], main_dir[0]))
        except Exception:
            v = p3 - p1
            theta = float(np.arctan2(v[1], v[0]))
        return center, theta

    @staticmethod
    def _solve_known_size_from_3_points(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
        a: float, b: float,
        center0: np.ndarray, theta0: float, max_residual: float,
    ) -> Optional[tuple[np.ndarray, float]]:
        """Solve (center, theta) from 3 points with known semi-axes (a, b)."""
        pts = np.vstack([p1, p2, p3])
        px, py = pts[:, 0], pts[:, 1]

        def residuals(params: np.ndarray) -> np.ndarray:
            cx, cy, theta = params
            u, v = transform_to_ellipse_frame(px, py, cx, cy, theta)
            return ((u / a) ** 2 + (v / b) ** 2 - 1.0).astype(np.float64)

        x0 = np.array([center0[0], center0[1], theta0], dtype=np.float64)
        result = least_squares(residuals, x0, method="lm")
        if not result.success:
            return None
        if float(np.linalg.norm(result.fun)) > max_residual:
            return None
        cx, cy, theta = result.x
        theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
        return np.array([cx, cy], dtype=np.float64), theta

    def fit(self, data: np.ndarray) -> list[Ellipse]:
        """Fit from minimal sample. Returns list of models (0 or 1)."""
        data = np.asarray(data, dtype=np.float64)
        out: list[Ellipse] = []
        if self.size is None:
            if data.shape[0] != 5:
                return []
            try:
                a, b, x0, y0, theta = self._fit_from_5_points(data)
            except (ValueError, linalg.LinAlgError):
                return []
            if not (a > 0 and b > 0 and np.isfinite([a, b, x0, y0, theta]).all()):
                return []
            params = np.array([a, b, x0, y0, theta], dtype=np.float64)
            out.append(Ellipse.from_params(params))
        else:
            if data.shape[0] != 3:
                return []
            p1, p2, p3 = data[0], data[1], data[2]
            center0, theta0 = self._initial_guess_from_3_points(p1, p2, p3)
            max_res = self.max_residual_sample
            if max_res is None:
                max_res = 0.01 * min(abs(self._a), abs(self._b))
            solved = self._solve_known_size_from_3_points(
                p1, p2, p3, self._a, self._b, center0, theta0, max_res
            )
            if solved is not None:
                center, theta = solved
                params = np.array([self._a, self._b, center[0], center[1], theta], dtype=np.float64)
                out.append(Ellipse.from_params(params))
        return out
