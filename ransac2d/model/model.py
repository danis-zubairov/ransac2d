from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..utils._numba_utils import (
    distance_to_circle,
    distance_to_ellipse,
    distance_to_line,
    distance_to_rectangle,
    transform_to_ellipse_frame,
    transform_to_rect_frame,
)


@dataclass
class BaseModel(ABC):
    """Base for fitted 2D shape models. Subclasses implement residuals() and from_params()."""

    @abstractmethod
    def residuals(self, points: np.ndarray) -> np.ndarray:
        """Per-point distance to the shape. Args: points (N, 2). Returns: (N,)."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.residuals() must be implemented by subclass"
        )

    @classmethod
    @abstractmethod
    def from_params(cls, params: np.ndarray) -> BaseModel:
        """Build model from parameter vector. Params layout is shape-specific."""
        raise NotImplementedError(
            f"{cls.__name__}.from_params() must be implemented by subclass"
        )


@dataclass
class Ellipse(BaseModel):
    """Ellipse: center (x0, y0), angle theta (rad), semi-axes (a, b)."""

    center: np.ndarray  # (2,)
    theta: float
    a: float
    b: float

    @property
    def size(self) -> tuple[float, float]:
        """(a, b) semi-axes for API compatibility."""
        return (self.a, self.b)

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """
        Per-point distance to the ellipse boundary. Uses numba if available.
        Args:
            points: (N, 2) array of 2D points.

        Returns:
            (N,) distances to the ellipse boundary.
        """
        px = points[:, 0]
        py = points[:, 1]
        cx, cy = float(self.center[0]), float(self.center[1])
        u, v = transform_to_ellipse_frame(px, py, cx, cy, self.theta)
        return distance_to_ellipse(u, v, self.a, self.b)

    @classmethod
    def from_params(cls, params: np.ndarray) -> Ellipse:
        """Build from (a, b, x0, y0, theta)."""
        theta = float((params[4] + np.pi) % (2.0 * np.pi) - np.pi)
        return cls(
            center=params[2:4].copy(),
            theta=theta,
            a=float(params[0]),
            b=float(params[1]),
        )


@dataclass
class Rectangle(BaseModel):
    """Rectangle: center (c_x, c_y), angle theta (rad), side lengths (a, b)."""

    center: np.ndarray  # (2,)
    theta: float
    a: float
    b: float

    @property
    def size(self) -> tuple[float, float]:
        """(a, b) side lengths for API compatibility."""
        return (self.a, self.b)

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """
        Per-point distance to the rectangle boundary. Uses numba if available.

        Args:
            points: (N, 2) array of 2D points.

        Returns:
            (N,) distances to the rectangle boundary.
        """
        px = points[:, 0]
        py = points[:, 1]
        cx, cy = float(self.center[0]), float(self.center[1])
        u, v = transform_to_rect_frame(px, py, cx, cy, self.theta)
        return distance_to_rectangle(u, v, self.a, self.b)

    def corners(self) -> np.ndarray:
        """Four corners in world coordinates, CCW."""
        cx, cy = float(self.center[0]), float(self.center[1])
        c, s = float(np.cos(self.theta)), float(np.sin(self.theta))
        a, b = self.a, self.b
        local = np.array(
            [
                [+a / 2.0, +b / 2.0],
                [-a / 2.0, +b / 2.0],
                [-a / 2.0, -b / 2.0],
                [+a / 2.0, -b / 2.0],
            ],
            dtype=np.float64,
        )
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        world = (R @ local.T).T
        world[:, 0] += cx
        world[:, 1] += cy
        return world

    @classmethod
    def from_params(cls, params: np.ndarray) -> Rectangle:
        """Build from (cx, cy, theta, a, b)."""
        theta = float((params[2] + np.pi) % (2.0 * np.pi) - np.pi)
        return cls(
            center=params[:2].copy(),
            theta=theta,
            a=float(params[3]),
            b=float(params[4]),
        )


@dataclass
class Circle(BaseModel):
    """Circle: center (x0, y0), radius r."""

    center: np.ndarray  # (2,)
    radius: float

    @property
    def size(self) -> float:
        """circle radius for API compatibility."""
        return self.radius

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """
        Per-point distance to the circle boundary.
        Args:
            points: (N, 2) array of 2D points.

        Returns:
            (N,) distances |‖p - c‖ - r|.
        """
        px = points[:, 0]
        py = points[:, 1]
        cx, cy = float(self.center[0]), float(self.center[1])
        return distance_to_circle(px, py, cx, cy, self.radius)

    @classmethod
    def from_params(cls, params: np.ndarray) -> Circle:
        """Build from (x0, y0, r)."""
        return cls(center=params[:2].copy(), radius=float(params[2]))


@dataclass
class Line(BaseModel):
    """Infinite line: implicit `a*x + b*y + c = 0` with `sqrt(a^2 + b^2) = 1`."""

    a: float
    b: float
    c: float

    def direction(self) -> np.ndarray:
        """Unit direction vector along the line (perpendicular to normal `(a, b)`)."""
        d = np.array([-self.b, self.a], dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return d
        return d / n

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """Per-point orthogonal distance to the line."""
        px = points[:, 0]
        py = points[:, 1]
        return distance_to_line(px, py, self.a, self.b, self.c)

    @classmethod
    def from_params(cls, params: np.ndarray) -> Line:
        """Build from `(a, b, c)`; re-normalizes so `sqrt(a^2 + b^2) = 1`."""
        a, b, c = float(params[0]), float(params[1]), float(params[2])
        norm = float(np.hypot(a, b))
        if norm < 1e-12:
            return cls(a=1.0, b=0.0, c=0.0)
        return cls(a=a / norm, b=b / norm, c=c / norm)
