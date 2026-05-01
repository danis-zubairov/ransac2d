"""
Demo: RANSAC line with visualization.

Run from repo root:

- ``python -m examples.demo_line``
- ``PYTHONPATH=src python examples/demo_line.py``
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import ensure_src_on_path
from examples.common import run_ransac_demo
from ransac2d import Line, LineEstimator, RANSACFitter


def generate_line_points(rng: np.random.Generator):
    true_line = Line.from_params(np.array([1.0, -0.7, -0.05], dtype=np.float64))

    n_inliers = 120
    x = rng.uniform(-0.8, 0.8, n_inliers)
    y = -(true_line.a * x + true_line.c) / true_line.b

    # Add isotropic noise around line points.
    line_points = np.column_stack([x, y]) + rng.normal(scale=0.01, size=(n_inliers, 2))

    n_outliers = 160
    outliers = rng.uniform(
        low=[-1.0, -1.0],
        high=[1.0, 1.0],
        size=(n_outliers, 2),
    )

    points = np.vstack([line_points, outliers])
    rng.shuffle(points)
    return points, true_line


def make_line_fitter(_: Line) -> RANSACFitter:
    n_iters = 2000
    return RANSACFitter(
        LineEstimator(),
        max_trials=n_iters,
        residual_threshold=0.03,
        min_inliers=60,
        random_state=0,
        n_jobs=8,
    )


def draw_line(ax: plt.Axes, model: Line, color="r", linestyle="-", linewidth=2, label=None):
    xs = np.array([-1.1, 1.1], dtype=np.float64)
    if abs(model.b) > 1e-10:
        ys = -(model.a * xs + model.c) / model.b
    else:
        # Vertical line fallback: x = -c / a
        x_val = -model.c / (model.a + 1e-12)
        xs = np.array([x_val, x_val], dtype=np.float64)
        ys = np.array([-1.1, 1.1], dtype=np.float64)
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def make_title(true_model: Line | None, est_model: Line | None) -> str:
    if true_model is not None and est_model is not None:
        return (
            "RANSAC line\n"
            f"True: {true_model.a:.3f}x + {true_model.b:.3f}y + {true_model.c:.3f} = 0\n"
            f"Est:  {est_model.a:.3f}x + {est_model.b:.3f}y + {est_model.c:.3f} = 0"
        )
    return "RANSAC line - no model found"


def on_fit(true_model: Line, est_model: Line | None) -> None:
    if est_model is not None:
        print(f"Fit:  a={est_model.a:.4f}, b={est_model.b:.4f}, c={est_model.c:.4f}")
        print(f"True: a={true_model.a:.4f}, b={true_model.b:.4f}, c={true_model.c:.4f}")
    else:
        print("RANSAC did not find a model.")


def main():
    ensure_src_on_path()
    run_ransac_demo(
        name="line",
        generate_points=generate_line_points,
        make_fitter=make_line_fitter,
        draw_true=lambda ax, m: draw_line(ax, m, color="g", linestyle="--", linewidth=2, label="True line"),
        draw_est=lambda ax, m: draw_line(ax, m, color="r", linestyle="-", linewidth=2, label="RANSAC line"),
        make_title=make_title,
        on_fit=on_fit,
        rng_seed=7,
        fig_size=(6.0, 6.0),
    )


if __name__ == "__main__":
    main()
