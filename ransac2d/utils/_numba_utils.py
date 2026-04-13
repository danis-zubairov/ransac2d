"""Numba-JIT helpers for vectorized transform and distance-to-shape.
njit compiles when numba is available, otherwise the same code runs
as plain NumPy.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(cache=True):
        """Stub when numba is not installed: returns the function unchanged."""
        return lambda f: f


# --- Transform to local frame (u, v) ---

@njit(cache=True)
def transform_to_rect_frame(px, py, cx, cy, theta):
    """Transform points to rectangle local frame. theta in radians."""
    c = np.cos(theta)
    s = np.sin(theta)
    dx = px - cx
    dy = py - cy
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return u, v


@njit(cache=True)
def transform_to_ellipse_frame(px, py, cx, cy, theta):
    """Transform points to ellipse local frame. theta in radians."""
    c = np.cos(theta)
    s = np.sin(theta)
    dx = px - cx
    dy = py - cy
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return u, v


# --- Distance to boundary ---

@njit(cache=True)
def distance_to_rectangle(u, v, a, b):
    """Euclidean distance from (u, v) to the boundary of [-a/2,a/2] x [-b/2,b/2]."""
    ha = a / 2.0
    hb = b / 2.0
    au = np.abs(u)
    av = np.abs(v)
    du = np.maximum(au - ha, 0.0)
    dv = np.maximum(av - hb, 0.0)
    inside = (au <= ha) & (av <= hb)
    dist_inside = np.minimum(ha - au, hb - av)
    dist_outside = np.sqrt(du * du + dv * dv)
    return np.where(inside, dist_inside, dist_outside)

@njit(cache=True)
def distance_to_ellipse(u, v, a, b):
    """Distance from (u, v) to ellipse u²/a² + v²/b² = 1. dist = |r - 1| * sqrt(a*b)."""
    r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return np.abs(r - 1.0) * np.sqrt(a * b)

@njit(cache=True)
def distance_to_circle(px, py, cx, cy, r):
    """Distance from (x, y) to circle (x-cx)^2 + (y-cy)^2 = r^2"""
    d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return np.abs(d - r)
