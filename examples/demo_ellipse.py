"""
Demo: RANSAC ellipse (5-point or known size) with visualization.

Run from repo root:

- ``python -m examples.demo_ellipse``
- ``PYTHONPATH=src python examples/demo_ellipse.py``
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import ensure_src_on_path
from examples.common import run_ransac_demo
from ransac2d import Ellipse, EllipseEstimator, RANSACFitter


def generate_ellipse_points(rng: np.random.Generator):
    true_a = 0.15
    true_b = 0.10
    true_x0 = 0.2
    true_y0 = -0.1
    true_theta = np.deg2rad(30.0)

    num_points = 100
    t = rng.uniform(0, 2 * np.pi, num_points)
    x = true_a * np.cos(t)
    y = true_b * np.sin(t)
    c, s = np.cos(true_theta), np.sin(true_theta)
    x_rot = c * x - s * y + true_x0
    y_rot = s * x + c * y + true_y0
    ellipse_points = np.column_stack([x_rot, y_rot]) + rng.normal(scale=0.001, size=(num_points, 2))

    num_outliers = 100
    outliers = rng.uniform(
        low=[-0.1, -0.2],
        high=[0.5, 0.3],
        size=(num_outliers, 2),
    )
    all_points = np.vstack([ellipse_points, outliers])

    true_center = np.array([true_x0, true_y0])
    true_model = Ellipse(center=true_center, theta=true_theta, a=true_a, b=true_b)
    return all_points, true_model


def make_ellipse_fitter(true_model: Ellipse) -> RANSACFitter:
    n_iters = 3000
    return RANSACFitter(
        EllipseEstimator(size=(true_model.a, true_model.b)),  # or EllipseEstimator()
        max_trials=n_iters,
        residual_threshold=0.004,
        min_inliers=40,
        random_state=0,
        n_jobs=8,
    )


def draw_ellipse(ax: plt.Axes, model: Ellipse, color="r", linestyle="-", linewidth=2, label=None):
    a, b = model.a, model.b
    cx, cy = model.center[0], model.center[1]
    theta = model.theta
    t = np.linspace(0, 2 * np.pi, 200)
    x = a * np.cos(t)
    y = b * np.sin(t)
    c, s = np.cos(theta), np.sin(theta)
    x_rot = c * x - s * y + cx
    y_rot = s * x + c * y + cy
    ax.plot(x_rot, y_rot, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def make_title(true_model: Ellipse | None, est_model: Ellipse | None) -> str:
    if true_model is not None and est_model is not None:
        true_theta_deg = np.rad2deg(true_model.theta)
        est_theta_deg = np.rad2deg(est_model.theta)
        return (
            "RANSAC ellipse\n"
            f"True: a={true_model.a:.3f}, b={true_model.b:.3f}, "
            f"center=({true_model.center[0]:.3f}, {true_model.center[1]:.3f}), θ={true_theta_deg:.1f}°\n"
            f"Est:  a={est_model.a:.3f}, b={est_model.b:.3f}, "
            f"center=({est_model.center[0]:.3f}, {est_model.center[1]:.3f}), θ={est_theta_deg:.1f}°"
        )
    return "RANSAC ellipse — no model found"


def on_fit(true_model: Ellipse, est_model: Ellipse | None) -> None:
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
        name="ellipse",
        generate_points=generate_ellipse_points,
        make_fitter=make_ellipse_fitter,
        draw_true=lambda ax, m: draw_ellipse(ax, m, color="g", linestyle="--", linewidth=2, label="True ellipse"),
        draw_est=lambda ax, m: draw_ellipse(ax, m, color="r", linestyle="-", linewidth=2, label="RANSAC ellipse"),
        make_title=make_title,
        on_fit=on_fit,
        rng_seed=42,
        fig_size=(8.0, 8.0),
    )


if __name__ == "__main__":
    main()
