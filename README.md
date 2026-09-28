
[![CI](https://github.com/danis-zubairov/ransac2d/actions/workflows/ci.yml/badge.svg)](https://github.com/danis-zubairov/ransac2d/actions/workflows/ci.yml) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ransac2d
---

Fit geometric shapes to noisy 2D point clouds with RANSAC.

<table border="0" cellpadding="6" cellspacing="0" align="center">
  <tr>
    <td align="center" valign="top" width="50%">
      <img src="docs/images/line.png" alt="Line RANSAC fit" width="420"/><br/>
      <b>Line</b>
    </td>
    <td align="center" valign="top" width="50%">
      <img src="docs/images/circle.png" alt="Circle RANSAC fit" width="420"/><br/>
      <b>Circle</b>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="50%">
      <img src="docs/images/ellipse.png" alt="Ellipse RANSAC fit" width="420"/><br/>
      <b>Ellipse</b>
    </td>
    <td align="center" valign="top" width="50%">
      <img src="docs/images/rectangle.png" alt="Rectangle RANSAC fit" width="420"/><br/>
      <b>Rectangle</b>
    </td>
  </tr>
</table>

## Features

- Robust fitting in the presence of outliers
- Line, circle, ellipse and rectangle fitting
- Known-radius / known-size fitting
- Parallel RANSAC
- Optional Numba acceleration


## Installation

Clone the repository and install the package:
```bash
git clone https://github.com/danis-zubairov/ransac2d.git
cd ransac2d
pip install .
```
For development:

```bash
pip install -e ".[dev,examples]"
```
With optional Numba acceleration:

```bash
pip install -e ".[fast]"
```

Requires `Python 3.9+`, `NumPy`, `SciPy`, and `joblib`.

## Quick start

Fit a circle to a 2D point cloud:

```python
import numpy as np

from ransac2d import CircleEstimator, RANSACFitter

# Example point cloud: (n_points, 2)
points = np.array([
  [0.35, 0.05],
  [0.27, 0.19],
  # ...
  [0.32, 0.13],
])

fitter = RANSACFitter(
    CircleEstimator(),
    max_trials=2000,
    residual_threshold=0.01,
    min_inliers=50,
    random_state=0,
)

fitter.fit(points)

circle = fitter.model_

if circle is not None:
    print("center:", circle.center)
    print("radius:", circle.radius)
    print("inliers:", fitter.inlier_mask_.sum())
```

`fitter.model_` contains the fitted circle, while `fitter.inlier_mask_` identifies the points classified as inliers.

See [`examples/`](examples/) for complete examples with visualization.

## Supported shapes

`ransac2d` supports the following 2D geometric primitives:

| Shape     | Minimal sample | Known parameters      |
| --------- | -------------: | --------------------- |
| Line      |       2 points | —                     |
| Circle    |       3 points | Radius                |
| Ellipse   |       5 points | Semi-axes `(a, b)`    |
| Rectangle |       5 points | Side lengths `(a, b)` |

For shapes with known size parameters, fewer points are required to generate a hypothesis.

## Examples

Complete examples with generated point clouds and visualization are available in [`examples/`](examples/):

* [Line](examples/demo_line.py)
* [Circle](examples/demo_circle.py)
* [Ellipse](examples/demo_ellipse.py)
* [Rectangle](examples/demo_rectangle.py)

Run an example from the repository root:

```bash
python -m examples.demo_circle
```

Replace `demo_circle` with `demo_line`, `demo_ellipse`, or `demo_rectangle` to try the other estimators.

## Parameters

`RANSACFitter` accepts the following main parameters:

| Parameter            | Description                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `max_trials`         | Maximum number of RANSAC trials in total. When `n_jobs > 1`, the trials are split between workers.        |
| `residual_threshold` | Maximum residual for a point to be considered an inlier. It uses the same units as the input coordinates. |
| `min_inliers`        | Minimum number of inliers required to accept a fitted model.                                              |
| `n_jobs`             | Number of parallel workers. Use `-1` to use all available CPU cores.                                      |
| `random_state`       | Seed used for reproducible sampling.                                                                      |

For example:

```python
fitter = RANSACFitter(
    CircleEstimator(),
    max_trials=3000,
    residual_threshold=0.005,
    min_inliers=40,
    n_jobs=-1,
    random_state=0,
)
```

## Output

After calling `fit(X)`, the fitted model and RANSAC results are available through:

| Attribute      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `model_`       | Best fitted geometric model, or `None` if no model satisfies `min_inliers`. |
| `inlier_mask_` | Boolean array indicating which input points are inliers.                    |
| `n_trials_`    | Number of RANSAC trials that were executed.                                 |

For example:

```python
fitter.fit(points)

if fitter.model_ is not None:
    model = fitter.model_
    inliers = points[fitter.inlier_mask_]
```

## Choosing parameters

The parameters that most directly affect fitting quality are
`residual_threshold` and `min_inliers`.

### `residual_threshold`

Choose a threshold based on the expected noise in your data. It has the same units as the input coordinates.

A threshold that is too small can reject valid noisy points. A threshold that is too large can classify outliers as inliers.

For example, if the typical positional noise is around `0.002`, a threshold around `0.005` may be a reasonable starting point.

### `min_inliers`

Set `min_inliers` to the minimum number of points you expect to belong to the shape.

If the point cloud contains only a small portion of the target shape, use a lower value. If the target shape should contain many points, a higher value can help reject accidental hypotheses.

### `max_trials`

Increase `max_trials` when the fraction of inliers is low or when a valid minimal sample is unlikely to contain only inliers. More trials improve the chance of finding a good model, at the cost of additional computation.

## Limitations

Runtime can be significant on low-end hardware, especially with large point clouds or a high `max_trials` value. Performance depends on the selected estimator, input size, and RANSAC parameters, so real-time suitability should be evaluated for the specific use case.

## License

[MIT](https://github.com/danis-zubairov/ransac2d/blob/master/LICENSE)
