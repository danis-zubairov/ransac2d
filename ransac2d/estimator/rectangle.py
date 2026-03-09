from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..utils._numba_utils import transform_to_rect_frame
from ..model import Rectangle
from .base import BaseEstimator


class RectangleEstimator(BaseEstimator):
    """
    Rectangle shape estimator: one class for unknown size (5-point, up to 20 hypotheses per sample)
    and known size (3-point, up to 4 hypotheses per sample).

    If size is None, 5-point (up to 20 hypotheses/sample).
    If size=(a, b), 3-point known-size (up to 4 hypotheses/sample).
    """

    def __init__(
        self,
        size: Optional[tuple[float, float]] = None,
        *,
        max_residual_sample: Optional[float] = None,
        lm_maxiter: int = 50,
    ):
        """Initialize the estimator.

        Args:
            size: None for 5-point fit; (a, b) for known side lengths (3-point).
            max_residual_sample: Max residual on minimal sample; None = 1% of scale or min(a,b).
            lm_maxiter: Max iterations for LM in 5-point solver.
        """
        self.size = size
        self.max_residual_sample = max_residual_sample
        self.lm_maxiter = lm_maxiter
        if size is not None:
            self._a, self._b = float(size[0]), float(size[1])

    @property
    def min_samples(self) -> int:
        """Minimum number of points (3 for known size, 5 for full)."""
        return 3 if self.size is not None else 5

    @staticmethod
    def _order_points_ccw(pts: np.ndarray) -> np.ndarray:
        """Order 2D points counter-clockwise around centroid."""
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return np.asarray(pts[np.argsort(angles)], dtype=np.float64)

    @staticmethod
    def _is_convex_pentagon(pts: np.ndarray) -> bool:
        """True if the 5 points form a convex pentagon."""
        p = np.vstack([pts, pts[0:2]])
        e1 = np.diff(p, axis=0)[:5]
        e2 = np.diff(p, axis=0)[1:6]
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        return bool(np.all(cross >= 0) or np.all(cross <= 0))

    @staticmethod
    def _initial_guess_from_5_points(pts_5: np.ndarray) -> np.ndarray:
        """Initial (center, theta, a, b) from 5 points."""
        pts_5 = np.asarray(pts_5, dtype=np.float64)
        center = np.mean(pts_5, axis=0)
        pts_c = pts_5 - center
        try:
            cov = np.cov(pts_c.T)
            vals, vecs = np.linalg.eig(cov)
            idx = np.argmax(np.real(vals))
            main_dir = np.real(vecs[:, idx])
            theta = float(np.arctan2(main_dir[1], main_dir[0]))
        except Exception:
            theta = 0.0
        px, py = pts_5[:, 0], pts_5[:, 1]
        u, v = transform_to_rect_frame(px, py, center[0], center[1], theta)
        a = max(float(np.ptp(u)), 1e-6)
        b = max(float(np.ptp(v)), 1e-6)
        return np.array([center[0], center[1], theta, a, b], dtype=np.float64)

    @staticmethod
    def _hypothesis_residuals_5pt(
        params: np.ndarray, points_5: np.ndarray, case_id: int, shift: int
    ) -> np.ndarray:
        cx, cy, theta, a, b = params[0], params[1], params[2], params[3], params[4]
        order = (np.arange(5) + shift) % 5
        pts = points_5[order]
        u, v = transform_to_rect_frame(pts[:, 0], pts[:, 1], cx, cy, theta)
        ha, hb = a / 2.0, b / 2.0
        u1, u2, u3, u4, u5 = u[0], u[1], u[2], u[3], u[4]
        v1, v2, v3, v4, v5 = v[0], v[1], v[2], v[3], v[4]
        if case_id == 0:
            return np.array([v1 - hb, v2 - hb, u3 + ha, v4 + hb, u5 - ha], dtype=np.float64)
        if case_id == 1:
            return np.array([u1 + ha, u2 + ha, v3 + hb, u4 - ha, v5 - hb], dtype=np.float64)
        if case_id == 2:
            return np.array([v1 + hb, v2 + hb, u3 - ha, v4 - hb, u5 + ha], dtype=np.float64)
        if case_id == 3:
            return np.array([u1 - ha, u2 - ha, v3 - hb, u4 + ha, v5 + hb], dtype=np.float64)
        raise ValueError(f"case_id must be 0..3, got {case_id}")

    @staticmethod
    def _solve_hypothesis_5pt(
        points_5: np.ndarray,
        init_params: np.ndarray,
        case_id: int,
        shift: int,
        max_nfev: int,
        max_residual: float,
    ) -> Optional[np.ndarray]:
        def fun(x: np.ndarray) -> np.ndarray:
            return RectangleEstimator._hypothesis_residuals_5pt(x, points_5, case_id, shift)

        res = least_squares(fun, init_params, method="lm", max_nfev=max_nfev)
        if not res.success:
            return None
        if float(np.linalg.norm(res.fun)) > max_residual:
            return None
        if res.x[3] <= 1e-6 or res.x[4] <= 1e-6:
            return None
        return res.x.copy()

    @staticmethod
    def _initial_guess_from_3_points(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> tuple[np.ndarray, float]:
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
        a: float, b: float, case_id: int,
        center0: np.ndarray, theta0: float, max_residual: float,
    ) -> Optional[tuple[np.ndarray, float]]:
        half_a, half_b = a / 2.0, b / 2.0
        pts = np.vstack([p1, p2, p3])
        px, py = pts[:, 0], pts[:, 1]

        def res_fn(params: np.ndarray) -> np.ndarray:
            cx, cy, theta = params
            u, v = transform_to_rect_frame(px, py, cx, cy, theta)
            u1, u2, u3 = u[0], u[1], u[2]
            v1, v2, v3 = v[0], v[1], v[2]
            if case_id == 1:
                return np.array([v1 + half_b, u2 - half_a, v3 - half_b], dtype=np.float64)
            if case_id == 2:
                return np.array([u1 + half_a, v2 + half_b, u3 - half_a], dtype=np.float64)
            if case_id == 3:
                return np.array([v1 - half_b, u2 + half_a, v3 + half_b], dtype=np.float64)
            if case_id == 4:
                return np.array([u1 - half_a, v2 - half_b, u3 + half_a], dtype=np.float64)
            raise ValueError(f"Unknown case_id: {case_id}")

        x0 = np.array([center0[0], center0[1], theta0], dtype=np.float64)
        result = least_squares(res_fn, x0, method="lm")
        if not result.success:
            return None
        if float(np.linalg.norm(result.fun)) > max_residual:
            return None
        cx, cy, theta = result.x
        theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
        return np.array([cx, cy], dtype=np.float64), theta

    def is_data_valid(self, sample: np.ndarray) -> bool:
        """Reject degenerate or non-convex samples."""
        if self.size is not None:
            if sample.shape[0] != 3:
                return False
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            return abs(v1[0] * v2[1] - v1[1] * v2[0]) >= 1e-6
        if sample.shape[0] != 5:
            return False
        ordered = self._order_points_ccw(np.asarray(sample, dtype=np.float64))
        return self._is_convex_pentagon(ordered)

    def fit(self, data: np.ndarray) -> list[Rectangle]:
        """Fit from minimal sample. Returns list of models (up to 20 or 4)."""
        data = np.asarray(data, dtype=np.float64)
        out: list[Rectangle] = []
        if self.size is None:
            if data.shape[0] != 5:
                return []
            sample_pts = self._order_points_ccw(data)
            init_params = self._initial_guess_from_5_points(sample_pts)
            max_res = self.max_residual_sample
            if max_res is None:
                scale = float(np.median(np.ptp(data, axis=0)))
                max_res = max(0.01 * scale, 1e-6)
            for case_id in range(4):
                for shift in range(5):
                    solved = self._solve_hypothesis_5pt(
                        sample_pts, init_params, case_id, shift,
                        max_nfev=self.lm_maxiter, max_residual=max_res,
                    )
                    if solved is not None:
                        out.append(Rectangle.from_params(solved))
        else:
            if data.shape[0] != 3:
                return []
            ordered = self._order_points_ccw(data)
            p1, p2, p3 = ordered[0], ordered[1], ordered[2]
            center0, theta0 = self._initial_guess_from_3_points(p1, p2, p3)
            max_res = self.max_residual_sample
            if max_res is None:
                max_res = 0.01 * min(abs(self._a), abs(self._b))
            for case_id in (1, 2, 3, 4):
                solved = self._solve_known_size_from_3_points(
                    p1, p2, p3, self._a, self._b, case_id, center0, theta0, max_res
                )
                if solved is not None:
                    center, theta = solved
                    params = np.array([center[0], center[1], theta, self._a, self._b], dtype=np.float64)
                    out.append(Rectangle.from_params(params))
        return out
