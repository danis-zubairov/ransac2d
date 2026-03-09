from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..model import BaseModel


class BaseEstimator(ABC):
    """
    Base class for 2D shape estimators used with RANSAC.

    Subclasses implement fit(data) returning a list of models;
    the RANSAC loop uses model.residuals(points) and keeps the best inlier set.
    """

    @property
    @abstractmethod
    def min_samples(self) -> int:
        """Minimum number of points required to fit this shape (e.g. 3 or 5)."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.min_samples() must be implemented by subclass"
        )

    @abstractmethod
    def fit(self, data: np.ndarray) -> list[BaseModel]:
        """Fit shape from a minimal sample.

        Args:
            data: (min_samples, 2) minimal sample of 2D points.

        Returns:
            List of model instances. Empty if no valid fit. Rectangle may
            return up to 20 (5pt) or 4 (known size); ellipse 0 or 1.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.fit() must be implemented by subclass"
        )

    def is_data_valid(self, sample: np.ndarray) -> bool:
        """Reject invalid minimal samples before fitting. Override as needed."""
        return True
