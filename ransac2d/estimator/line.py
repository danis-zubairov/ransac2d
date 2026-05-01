from __future__ import annotations

import numpy as np

from ..model import Line
from .base import BaseEstimator


class LineEstimator(BaseEstimator):
    """Line from two points: normalized implicit `a*x + b*y + c = 0`."""

    @property
    def min_samples(self) -> int:
        return 2

    def is_data_valid(self, sample: np.ndarray) -> bool:
        sample = np.asarray(sample, dtype=np.float64)
        if sample.shape[0] != 2:
            return False
        return float(np.linalg.norm(sample[1] - sample[0])) > 1e-8

    def fit(self, data: np.ndarray) -> list[Line]:
        """Fit from two distinct points. Returns one `Line` or empty list."""
        data = np.asarray(data, dtype=np.float64)
        if data.shape[0] != 2:
            return []
        p1, p2 = data[0], data[1]
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = float(np.hypot(a, b))
        if norm < 1e-12:
            return []
        return [Line(a=a / norm, b=b / norm, c=c / norm)]
