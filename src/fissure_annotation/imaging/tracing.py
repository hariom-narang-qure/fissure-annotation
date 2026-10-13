"""Trace a fissure between hand-picked waypoints along the bright ridge of the slice.

Steps, and why:
- Ridge strength is minus the Laplacian of the Gaussian-smoothed image (sigma 1.2 px): a thin bright line has a strongly negative second derivative across it.
- Path cost is 1 / (1 + ridge / 4.25)^2 with the ridge capped at 85: strong ridges are cheap, and the cap stops a bright vessel from being worth any detour.
  - 85 and 4.25 are 200 HU and 10 HU in the 8-bit levels of the 600 HU lung window (see slices.py).
- The waypoints are fixed anchors. The path between consecutive anchors is the minimal-cost path, so the line follows the ridge and still passes through every waypoint.
"""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Bool, Float, UInt8
from scipy import ndimage
from skimage.graph import route_through_array

RIDGE_SIGMA_PX = 1.2
RIDGE_CAP = 85.0
RIDGE_SCALE = 4.25

Point = tuple[int, int]  # (x, y) = (column, row)


def ridge_strength(image: UInt8[np.ndarray, "h w"]) -> Float[np.ndarray, "h w"]:
    pixels = image.astype(float)
    laplacian = ndimage.gaussian_filter(
        pixels, RIDGE_SIGMA_PX, order=(2, 0)
    ) + ndimage.gaussian_filter(pixels, RIDGE_SIGMA_PX, order=(0, 2))
    return np.clip(-laplacian, 0, None)


def path_cost(image: UInt8[np.ndarray, "h w"]) -> Float[np.ndarray, "h w"]:
    return 1 / (1 + np.clip(ridge_strength(image), 0, RIDGE_CAP) / RIDGE_SCALE) ** 2


def trace_between(
    cost: Float[np.ndarray, "h w"], start: Point, end: Point
) -> list[tuple[int, int]]:
    "pixels as (row, column)"
    path, _ = route_through_array(
        cost,
        (start[1], start[0]),
        (end[1], end[0]),
        fully_connected=True,
        geometric=True,
    )
    return path


def trace_mask(
    image: UInt8[np.ndarray, "h w"], fissures: Sequence[Sequence[Point]]
) -> Bool[np.ndarray, "h w"]:
    cost = path_cost(image)
    mask = np.zeros(image.shape, dtype=bool)
    for waypoints in fissures:
        for start, end in zip(waypoints, waypoints[1:]):
            rows, columns = zip(*trace_between(cost, start, end))
            mask[rows, columns] = True
    return mask
