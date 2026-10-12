"""Proxy score (NOT accuracy): the lobe-segmentation boundaries stand in for fissures.

- `dist_mm`: mean distance from the model's polyline to the nearest lobe boundary.
- `within5`: share of polyline points within 5 mm of a boundary.
- `recall5`: share of boundary pixels within 5 mm of any model polyline.
- A variant that was shown the boundaries (C, D, E, and every agent) is not scored independently of its input.
"""
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from fissure_annotation.lines import DrawableFissure
from fissure_annotation.slicing import boundary_mask
from fissure_annotation.variants import to_slice_pixels


def polyline_points(pts: list[tuple[float, float]], step: float = 2.0) -> np.ndarray:
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 1)
        out += [(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n) for t in range(n + 1)]
    return np.array(out)


def proxy_scores(view, fissures: Sequence[DrawableFissure], prepared) -> dict:
    bmask = boundary_mask(view.lobes)
    if not bmask.any() or not fissures:
        return {"dist_mm": None, "within5": None, "recall5": None}
    dist_to_boundary = ndimage.distance_transform_edt(~bmask) * view.mm_per_px
    h, w = bmask.shape
    pts = np.vstack([polyline_points([to_slice_pixels(p, prepared) for p in f.waypoints]) for f in fissures])
    ix = np.clip(pts.round().astype(int), 0, [w - 1, h - 1])
    d = dist_to_boundary[ix[:, 1], ix[:, 0]]
    line = np.zeros_like(bmask)
    line[ix[:, 1], ix[:, 0]] = True
    dist_to_line = ndimage.distance_transform_edt(~line) * view.mm_per_px
    return {"dist_mm": round(float(d.mean()), 1), "within5": round(float((d <= 5).mean()), 2),
            "recall5": round(float((dist_to_line[bmask] <= 5).mean()), 2)}
