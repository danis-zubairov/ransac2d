from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ransac2d import RANSACFitter

GenerateFn = Callable[[np.random.Generator], tuple[np.ndarray, Any]]
FitterFn = Callable[[Any], RANSACFitter]
DrawFn = Callable[[Axes, Any], None]
TitleFn = Callable[[Any | None, Any | None], str]
OnFitFn = Callable[[Any, Any | None], None]


@dataclass
class DemoResult:
    points: np.ndarray
    true_model: Any
    est_model: Any | None
    fitter: RANSACFitter
    elapsed: float


def run_ransac_demo(
    name: str,
    generate_points: GenerateFn,
    make_fitter: FitterFn,
    draw_true: DrawFn,
    draw_est: DrawFn,
    make_title: TitleFn,
    on_fit: OnFitFn | None = None,
    rng_seed: int = 42,
    fig_size: tuple[float, float] = (6.0, 6.0),
) -> DemoResult:
    """Common driver for 2D RANSAC demos."""
    rng = np.random.default_rng(rng_seed)

    points, true_model = generate_points(rng)

    print(f"Running RANSAC {name}...")
    fitter = make_fitter(true_model)
    t0 = time.perf_counter()
    fitter.fit(points)
    elapsed = time.perf_counter() - t0
    n_iters = getattr(fitter, "max_trials", None)
    if n_iters:
        print(f"  Time: {elapsed:.2f} s  ({1000 * elapsed / n_iters:.1f} ms per iter)")
    else:
        print(f"  Time: {elapsed:.2f} s")

    est_model = fitter.model_

    if on_fit is not None:
        on_fit(true_model, est_model)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(points[:, 0], points[:, 1], s=10, c="gray", alpha=0.5, label="All points")

    draw_true(ax, true_model)

    if est_model is not None:
        draw_est(ax, est_model)
        if fitter.inlier_mask_ is not None:
            inliers = points[fitter.inlier_mask_]
            ax.scatter(inliers[:, 0], inliers[:, 1], s=20, c="blue", alpha=0.7, label="Inliers")
    ax.set_title(make_title(true_model, est_model))
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return DemoResult(points=points, true_model=true_model, est_model=est_model, fitter=fitter, elapsed=elapsed)
