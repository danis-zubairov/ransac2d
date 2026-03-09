"""
Demo: RANSAC circle (unknown or known radius) with visualization.

Run from repo root:

- ``python -m examples.demo_circle``
- ``PYTHONPATH=src python examples/demo_circle.py``
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import ensure_src_on_path
from examples.common import run_ransac_demo
from ransac2d import Circle, CircleEstimator, RANSACFitter


def generate_circle_points(rng: np.random.Generator):
    true_center = np.array([0.2, -0.1], dtype=np.float64)
    true_radius = 0.25

    num_points = 80
    angles = rng.uniform(0, 2 * np.pi, num_points)
    x = true_center[0] + true_radius * np.cos(angles)
    y = true_center[1] + true_radius * np.sin(angles)
    circle_points = np.column_stack([x, y]) + rng.normal(scale=0.003, size=(num_points, 2))

    num_outliers = 100
    outliers = rng.uniform(
        low=[-0.2, -0.3],
        high=[0.6, 0.4],
        size=(num_outliers, 2),
    )
    all_points = np.vstack([circle_points, outliers])

    true_model = Circle(center=true_center, radius=true_radius)
    return all_points, true_model


def make_circle_fitter(true_model: Circle) -> RANSACFitter:
    n_iters = 3000
    return RANSACFitter(
        CircleEstimator(radius=true_model.radius),  # or CircleEstimator()
        max_trials=n_iters,
        residual_threshold=0.004,
        min_inliers=40,
        random_state=0,
        n_jobs=8,
    )


def draw_circle(ax: plt.Axes, model: Circle, color="r", linestyle="-", linewidth=2, label=None):
    cx, cy = model.center
    r = model.radius
    t = np.linspace(0, 2 * np.pi, 200)
    x = cx + r * np.cos(t)
    y = cy + r * np.sin(t)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def make_title(true_model: Circle | None, est_model: Circle | None) -> str:
    if true_model is not None and est_model is not None:
        return (
            "RANSAC circle\n"
            f"True: r={true_model.radius:.3f}, center=({true_model.center[0]:.3f}, {true_model.center[1]:.3f})\n"
            f"Est:  r={est_model.radius:.3f}, center=({est_model.center[0]:.3f}, {est_model.center[1]:.3f})"
        )
    return "RANSAC circle — no model found"


def on_fit(true_model: Circle, est_model: Circle | None) -> None:
    if est_model is not None:
        print(
            f"Fit: center=({est_model.center[0]:.3f}, {est_model.center[1]:.3f}), "
            f"r={est_model.radius:.3f}"
        )
        print(
            f"True: center=({true_model.center[0]:.3f}, {true_model.center[1]:.3f}), "
            f"r={true_model.radius:.3f}"
        )
    else:
        print("RANSAC did not find a model.")


def main():
    ensure_src_on_path()
    run_ransac_demo(
        name="circle",
        generate_points=generate_circle_points,
        make_fitter=make_circle_fitter,
        draw_true=lambda ax, m: draw_circle(ax, m, color="g", linestyle="--", linewidth=2, label="True circle"),
        draw_est=lambda ax, m: draw_circle(ax, m, color="r", linestyle="-", linewidth=2, label="RANSAC circle"),
        make_title=make_title,
        on_fit=on_fit,
        rng_seed=123,
        fig_size=(6.0, 6.0),
    )


if __name__ == "__main__":
    main()
