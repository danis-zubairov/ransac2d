from __future__ import annotations

from typing import Any, Optional

import numpy as np
from joblib import Parallel, delayed


class RANSACFitter:
    """RANSAC fitter for 2D shapes. fit(X) then model_, inlier_mask_.

    Trials are split evenly across n_jobs (n_trials_per_job = max_trials // n_jobs).
    """

    def __init__(
        self,
        estimator: Any,
        *,
        max_trials: int = 1000,
        residual_threshold: float = 0.005,
        min_inliers: int = 10,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ):
        """Initialize the fitter.

        Args:
            estimator: Shape estimator with fit(sample) -> list of models.
            max_trials: Maximum RANSAC iterations (total across all jobs).
            residual_threshold: Max distance for a point to be an inlier.
            min_inliers: Minimum inliers to accept a model.
            n_jobs: Number of parallel jobs. -1 = all cores. 1 = no parallelism.
            random_state: Random seed (each job gets seed + job_index).
        """
        self.estimator = estimator
        self.max_trials = max_trials
        self.residual_threshold = residual_threshold
        self.min_inliers = min_inliers
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model_: Optional[Any] = None
        self.inlier_mask_: Optional[np.ndarray] = None
        self.n_trials_: int = 0

    def fit(self, X: np.ndarray) -> RANSACFitter:
        """Fit the shape to 2D points using RANSAC.

        Args:
            X: (n_samples, 2) array of 2D points.

        Returns:
            self (model_, inlier_mask_, n_trials_ set).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be of shape (n_samples, 2)")
        n = X.shape[0]
        min_s = self.estimator.min_samples
        if n < min_s:
            raise ValueError(f"Need at least {min_s} points, got {n}")

        n_jobs = self.n_jobs
        if n_jobs <= 0:
            import os
            n_jobs = max(1, os.cpu_count() or 1)
        n_jobs = min(n_jobs, self.max_trials)
        n_trials_per_job = self.max_trials // n_jobs
        if n_trials_per_job < 1:
            n_jobs = 1
            n_trials_per_job = self.max_trials

        if self.random_state is not None:
            seeds = [self.random_state + j for j in range(n_jobs)]
        else:
            seeds = np.random.randint(0, np.iinfo(np.int32).max, size=n_jobs)

        if n_jobs == 1:
            best_model, best_mask, _ = self._run_batch_trials(
                X,
                n_trials_per_job,
                seeds[0],
                self.residual_threshold,
                self.min_inliers,
            )
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._run_batch_trials)(
                    X,
                    n_trials_per_job,
                    seed,
                    self.residual_threshold,
                    self.min_inliers,
                )
                for seed in seeds
            )
            best_model = None
            best_mask = None
            best_inliers = -1
            for model, mask, _ in results:
                if mask is not None:
                    n_in = int(np.sum(mask))
                    if n_in > best_inliers:
                        best_inliers = n_in
                        best_model = model
                        best_mask = mask

        self.model_ = best_model
        self.inlier_mask_ = best_mask
        self.n_trials_ = n_jobs * n_trials_per_job
        return self

    def _run_batch_trials(
        self,
        points: np.ndarray,
        n_trials: int,
        seed: Optional[int],
        residual_threshold: float,
        min_inliers: int,
    ) -> tuple[Optional[Any], Optional[np.ndarray], int]:
        """Run a fixed number of RANSAC trials.

        Returns:
            (best_model, best_mask, n_iters). Each trial may produce multiple
            hypotheses (e.g. rectangle 20 or 4); all evaluated via model.residuals.
        """
        rng = np.random.default_rng(seed)
        n_points = points.shape[0]
        min_samples = self.estimator.min_samples

        best_model = None
        best_inliers = -1
        best_mask: Optional[np.ndarray] = None

        for _ in range(n_trials):
            idx = rng.choice(n_points, size=min_samples, replace=False)
            sample = points[idx]

            if not self.estimator.is_data_valid(sample):
                continue

            models: list[Any] = self.estimator.fit(sample)
            for model in models:
                residuals = model.residuals(points)
                mask = residuals <= residual_threshold
                n_in = int(np.sum(mask))

                if n_in > best_inliers and n_in >= min_inliers:
                    best_inliers = n_in
                    best_model = model
                    best_mask = mask

        return best_model, best_mask, n_trials
