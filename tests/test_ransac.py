"""Basic tests for ransac2d"""

import numpy as np
import pytest

from ransac2d import (
    Circle,
    CircleEstimator,
    Ellipse,
    EllipseEstimator,
    LineEstimator,
    RANSACFitter,
    Rectangle,
    RectangleEstimator,
)


def _rectangle_points(n: int, center, theta: float, a: float, b: float, noise: float = 0.02, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0, 1, max(4, n // 4), endpoint=False)
    u = np.r_[np.full_like(t, a / 2), np.linspace(a / 2, -a / 2, len(t)), np.full_like(t, -a / 2), np.linspace(-a / 2, a / 2, len(t))]
    v = np.r_[np.linspace(-b / 2, b / 2, len(t)), np.full_like(t, b / 2), np.linspace(b / 2, -b / 2, len(t)), np.full_like(t, -b / 2)]
    u = u[:n]
    v = v[:n]
    c, s = np.cos(theta), np.sin(theta)
    x = c * u - s * v + center[0]
    y = s * u + c * v + center[1]
    x += rng.normal(0, noise, n)
    y += rng.normal(0, noise, n)
    return np.column_stack([x, y])

def _ellipse_points(n: int, a: float, b: float, x0: float, y0: float, theta: float, noise: float = 0.02, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    u = a * np.cos(t)
    v = b * np.sin(t)
    c, s = np.cos(theta), np.sin(theta)
    x = c * u - s * v + x0
    y = s * u + c * v + y0
    x += rng.normal(0, noise, n)
    y += rng.normal(0, noise, n)
    return np.column_stack([x, y])

def _circle_points(n: int, r: float, x0: float, y0: float, noise: float = 0.02, rng=None):
    rng = rng or np.random.default_rng(47)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)
    x += rng.normal(0, noise, n)
    y += rng.normal(0, noise, n)
    return np.column_stack([x, y])

def test_fit_rectangle():
    rng = np.random.default_rng(42)
    center = np.array([1.0, 2.0])
    theta = 0.3
    a, b = 1.5, 0.8
    points = _rectangle_points(160, center, theta, a, b, noise=0.03, rng=rng)
    fitter = RANSACFitter(
        RectangleEstimator(),
        max_trials=200,
        residual_threshold=0.05,
        min_inliers=30,
        random_state=123,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert isinstance(fitter.model_, Rectangle)
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >= 30

def test_fit_rectangle_known_size():
    rng = np.random.default_rng(44)
    center = np.array([0.0, 0.0])
    theta = 0.2
    a, b = 1.0, 0.6
    points = _rectangle_points(100, center, theta, a, b, noise=0.02, rng=rng)
    fitter = RANSACFitter(
        RectangleEstimator(size=(a, b)),
        max_trials=200,
        residual_threshold=0.04,
        min_inliers=20,
        random_state=789,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert isinstance(fitter.model_, Rectangle)
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >=20
    assert fitter.model_.a == a
    assert fitter.model_.b == b

def test_fit_ellipse():
    rng = np.random.default_rng(43)
    a, b, x0, y0, theta = 2.0, 1.0, 0.5, -0.5, 0.4
    points = _ellipse_points(120, a, b, x0, y0, theta, noise=0.03, rng=rng)
    fitter = RANSACFitter(
        EllipseEstimator(),
        max_trials=300,
        residual_threshold=0.08,
        min_inliers=25,
        random_state=456,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert isinstance(fitter.model_, Ellipse)
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >= 25

def test_fit_ellipse_known_size():
    rng = np.random.default_rng(45)
    a, b, x0, y0, theta = 1.5, 0.7, 0.0, 0.0, 0.1
    points = _ellipse_points(80, a, b, x0, y0, theta, noise=0.02, rng=rng)
    fitter = RANSACFitter(
        EllipseEstimator(size=(a, b)),
        max_trials=150,
        residual_threshold=0.05,
        min_inliers=15,
        random_state=101,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert isinstance(fitter.model_, Ellipse)
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >= 15
    assert fitter.model_.a == a
    assert fitter.model_.b == b

def test_fit_circle():
    rng = np.random.default_rng(88)
    r, x0, y0 = 0.3, 0.1, 0.2
    points = _circle_points(120, r, x0, y0, noise=0.03, rng=rng)
    fitter = RANSACFitter(
        CircleEstimator(),
        max_trials=300,
        residual_threshold=0.01,
        min_inliers=25,
        random_state=99,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert isinstance(fitter.model_, Circle)
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >= 25

def _line_points_inclined(n: int, rng=None):
    """Points near y = 0.5*x + 0.1 with small noise."""
    rng = rng or np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, n)
    y = 0.5 * x + 0.1 + rng.normal(0, 0.02, n)
    return np.column_stack([x, y])


def _line_points_near_vertical(n: int, rng=None):
    """Points near x = 0.3 with small noise (nearly vertical edge)."""
    rng = rng or np.random.default_rng(2)
    x = 0.3 + rng.normal(0, 0.015, n)
    y = rng.uniform(-1.0, 1.0, n)
    return np.column_stack([x, y])


def test_fit_line_inclined():
    rng = np.random.default_rng(11)
    points = _line_points_inclined(200, rng=rng)
    fitter = RANSACFitter(
        LineEstimator(),
        max_trials=400,
        residual_threshold=0.05,
        min_inliers=40,
        random_state=55,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >= 40
    a, b, c = fitter.model_.a, fitter.model_.b, fitter.model_.c
    assert abs(np.hypot(a, b) - 1.0) < 1e-9
    dists = np.abs(points @ np.array([a, b]) + c)
    assert np.all(dists[fitter.inlier_mask_] <= 0.05 + 1e-9)


def test_fit_line_near_vertical():
    rng = np.random.default_rng(12)
    points = _line_points_near_vertical(180, rng=rng)
    fitter = RANSACFitter(
        LineEstimator(),
        max_trials=400,
        residual_threshold=0.05,
        min_inliers=35,
        random_state=56,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >= 35
    a, b, c = fitter.model_.a, fitter.model_.b, fitter.model_.c
    assert abs(np.hypot(a, b) - 1.0) < 1e-9
    dists = np.abs(points @ np.array([a, b]) + c)
    assert np.all(dists[fitter.inlier_mask_] <= 0.05 + 1e-9)


def test_fit_circle_known_size():
    rng = np.random.default_rng(97)
    r, x0, y0 = 0.5, 0.4, 0.2
    points = _circle_points(120, r, x0, y0, noise=0.03, rng=rng)
    fitter = RANSACFitter(
        CircleEstimator(radius=r),
        max_trials=300,
        residual_threshold=0.01,
        min_inliers=25,
        random_state=99,
    )
    fitter.fit(points)
    assert fitter.model_ is not None
    assert isinstance(fitter.model_, Circle)
    assert fitter.inlier_mask_ is not None
    assert np.sum(fitter.inlier_mask_) >= 25
    assert fitter.model_.radius  == r

def test_parallel_jobs():
    rng = np.random.default_rng(46)
    points = _rectangle_points(80, np.array([0.0, 0.0]), 0.0, 1.0, 0.5, noise=0.02, rng=rng)
    fitter = RANSACFitter(
        RectangleEstimator(),
        max_trials=100,
        residual_threshold=0.05,
        min_inliers=15,
        n_jobs=2,
        random_state=202,
    )
    fitter.fit(points)
    # n_trials_ = n_jobs * (max_trials // n_jobs)
    assert fitter.n_trials_ == 100

def test_fit_raises_on_bad_input():
    fitter = RANSACFitter(RectangleEstimator())
    with pytest.raises(ValueError, match="shape"):
        fitter.fit(np.random.randn(10, 3))
    with pytest.raises(ValueError, match="at least 5"):
        fitter.fit(np.random.randn(3, 2))
