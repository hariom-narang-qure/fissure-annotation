"""Lung lobe ids, and distances to a lobe mask.

- Lobe ids are the TotalSegmentator ids of the `total` task.
"""

import numpy as np
from jaxtyping import Bool, Float
from scipy import ndimage

LEFT_UPPER, LEFT_LOWER, RIGHT_UPPER, RIGHT_MIDDLE, RIGHT_LOWER = 10, 11, 12, 13, 14

Spacing = tuple[float, float, float]


def distance_to_mask_mm(
    mask: Bool[np.ndarray, "x y z"], spacing_mm: Spacing
) -> Float[np.ndarray, "x y z"]:
    return ndimage.distance_transform_edt(~mask, sampling=spacing_mm)
