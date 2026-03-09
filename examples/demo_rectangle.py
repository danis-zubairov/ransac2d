"""
Demo: RANSAC rectangle (unknown size) with visualization.

Run from repo root:

- ``python -m examples.demo_rectangle``
- ``PYTHONPATH=src python examples/demo_rectangle.py``
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import ensure_src_on_path
from examples.common import run_ransac_demo
from ransac2d import RANSACFitter, Rectangle, RectangleEstimator


def sample_rectangle_edges(model: Rectangle, n_per_edge: int = 50, noise: float = 0.002, rng=None):
    rng = rng or np.random.default_rng(42)
    corners = model.corners()
    edge_pts = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        t = rng.random(n_per_edge)
        pts = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
        pts += rng.normal(scale=noise, size=pts.shape)
        edge_pts.append(pts)
    return np.vstack(edge_pts)


def generate_rectangle_points(rng: np.random.Generator):
    noise_scale = 0.002
    n_per_edge = 100

    true_center = np.array([-0.05, 0.12], dtype=np.float64)
    true_theta = np.deg2rad(-25.0)
    true_a, true_b = 0.30, 0.40
    true_model = Rectangle(center=true_center.copy(), theta=true_theta, a=true_a, b=true_b)
    edge_pts = sample_rectangle_edges(true_model, n_per_edge, noise_scale, rng)

    n_out = 800
    outliers = rng.uniform(
        low=[-0.35, -0.4],
        high=[0.45, 0.5],
        size=(n_out, 2),
    )
    points = np.vstack([edge_pts, outliers])
    rng.shuffle(points)
    return points, true_model


def make_rectangle_fitter(true_model: Rectangle) -> RANSACFitter:
    n_iters = 5000
    return RANSACFitter(
        RectangleEstimator(size=(true_model.a, true_model.b)),  # or RectangleEstimator()
        max_trials=n_iters,
        residual_threshold=0.001,
        min_inliers=35,
        n_jobs=8,
        random_state=0,
    )


def draw_rectangle(ax: plt.Axes, model: Rectangle, style: str, label: str):
    corners = model.corners()
    poly = np.vstack([corners, corners[0]])
    if style == "true":
        ax.plot(poly[:, 0], poly[:, 1], "g--", lw=2, label=label)
    else:
        ax.plot(poly[:, 0], poly[:, 1], "r-", lw=2, label=label)


def make_title(true_model: Rectangle | None, est_model: Rectangle | None) -> str:
    if true_model is not None and est_model is not None:
        true_theta_deg = np.rad2deg(true_model.theta)
        est_theta_deg = np.rad2deg(est_model.theta)
        return (
            "Rectangle RANSAC 5 params\n"
            f"True: c=({true_model.center[0]:.3f},{true_model.center[1]:.3f}), θ={true_theta_deg:.1f}° | "
            f"Est: c=({est_model.center[0]:.3f},{est_model.center[1]:.3f}), θ={est_theta_deg:.1f}°"
        )
    return "Rectangle RANSAC 5 params — no model found"


def on_fit(true_model: Rectangle, est_model: Rectangle | None) -> None:
    if est_model is not None:
        c = est_model.center
        print(
            f"Fit: center=({c[0]:.3f},{c[1]:.3f}), "
            f"θ={np.rad2deg(est_model.theta):.2f}°, a={est_model.a:.4f}, b={est_model.b:.4f}"
        )
        tc = true_model.center
        print(
            f"True: center=({tc[0]:.3f},{tc[1]:.3f}), "
            f"θ={np.rad2deg(true_model.theta):.2f}°, a={true_model.a:.3f}, b={true_model.b:.3f}"
        )
    else:
        print("RANSAC did not find a model.")


def main():
    ensure_src_on_path()
    run_ransac_demo(
        name="rectangle",
        generate_points=generate_rectangle_points,
        make_fitter=make_rectangle_fitter,
        draw_true=lambda ax, m: draw_rectangle(ax, m, style="true", label="True rectangle"),
        draw_est=lambda ax, m: draw_rectangle(ax, m, style="est", label="RANSAC fit"),
        make_title=make_title,
        on_fit=on_fit,
        rng_seed=42,
        fig_size=(6.0, 6.0),
    )


if __name__ == "__main__":
    main()
