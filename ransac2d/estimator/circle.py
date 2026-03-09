from __future__ import annotations

from typing import Optional

import numpy as np

from ..model import Circle
from .base import BaseEstimator


class CircleEstimator(BaseEstimator):
    """
    Circle shape estimator: unknown radius (3-point) and known radius (2-point).

    If radius is None, fits center and radius from 3 points.
    If radius is given, fits center from 2 points and returns up to two hypotheses.
    """

    def __init__(
        self,
        radius: Optional[float] = None,
        *,
        max_residual_sample: Optional[float] = None,
    ):
        """Initialize the estimator.

        Args:
            radius: Known radius. If None, fit radius as well.
            max_residual_sample: Max residual on minimal sample; if None, 1% of radius
                (for known-radius mode) or of robust scale of sample.
        """
        self.radius = radius
        self.max_residual_sample = max_residual_sample

    @property
    def min_samples(self) -> int:
        """Minimum number of points (2 for known radius, 3 for unknown)."""
        return 2 if self.radius is not None else 3

    def is_data_valid(self, sample: np.ndarray) -> bool:
        """Reject degenerate samples."""
        sample = np.asarray(sample, dtype=np.float64)
        if self.radius is not None:
            if sample.shape[0] != 2:
                return False
            # Points must not coincide.
            return np.linalg.norm(sample[1] - sample[0]) > 1e-8
        if sample.shape[0] != 3:
            return False
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        # Area of triangle should be non-zero.
        return abs(v1[0] * v2[1] - v1[1] * v2[0]) >= 1e-10

    @staticmethod
    def _fit_from_3_points(pts: np.ndarray) -> Optional[Circle]:
        """Fit circle (center, radius) from 3 non-collinear points.

        Uses system: (x^2 + y^2, x, y) @ (A, D, E)^T = -1, then

            x0 = -D / (2A), y0 = -E / (2A),
            r = (1 / |2A|) * sqrt(D^2 + E^2 - 4A).

        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] != 3:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        M = np.empty((3, 3), dtype=np.float64)
        M[:, 0] = x**2 + y**2
        M[:, 1] = x
        M[:, 2] = y
        b = -np.ones(3, dtype=np.float64)
        try:
            A, D, E = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite([A, D, E])):
            return None
        if abs(A) < 1e-12:
            return None
        x0 = -D / (2.0 * A)
        y0 = -E / (2.0 * A)
        disc = (D * D + E * E - 4.0 * A)
        if disc < 0:
            return None
        r = 1 / (2 * abs(A)) * float(np.sqrt(disc))
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(r) and r > 0):
            return None
        return Circle(center=np.array([x0, y0], dtype=np.float64), radius=float(r))

    @staticmethod
    def _fit_known_radius_from_2_points(
        p1: np.ndarray, p2: np.ndarray, radius: float
    ) -> list[Circle]:
        """
        Fit circle centers with known radius from 2 points.

        Returns up to two circles whose boundary passes through p1 and p2.
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        d_vec = p2 - p1
        d = float(np.linalg.norm(d_vec))
        if d < 1e-12 or radius <= 0.0:
            return []
        if d > 2.0 * radius + 1e-9:
            # No circle of given radius passes through both points.
            return []
        mid = 0.5 * (p1 + p2)
        # Unit normal to segment.
        n = np.array([-d_vec[1], d_vec[0]], dtype=np.float64) / d
        h_sq = radius * radius - (d * 0.5) ** 2
        if h_sq < 0.0:
            return []
        h = float(np.sqrt(max(h_sq, 0.0)))
        centers = [mid + h * n, mid - h * n] if h > 0 else [mid]
        models: list[Circle] = []
        for c in centers:
            models.append(Circle(center=np.asarray(c, dtype=np.float64), radius=float(radius)))
        return models

    def fit(self, data: np.ndarray) -> list[Circle]:
        """Fit from minimal sample. Returns list of Circle models."""
        data = np.asarray(data, dtype=np.float64)
        out: list[Circle] = []
        if self.radius is None:
            if data.shape[0] != 3:
                return []
            model = self._fit_from_3_points(data)
            if model is None:
                return []
            out.append(model)
        else:
            if data.shape[0] != 2:
                return []
            models = self._fit_known_radius_from_2_points(data[0], data[1], float(self.radius))
            if not models:
                return []
            out.extend(models)
        return out
