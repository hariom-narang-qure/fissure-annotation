"""Lung lobe ids, and distances to a lobe mask.

- Lobe ids are the TotalSegmentator ids of the `total` task.
- A fissure is the interface between two lobes. `FISSURE_LOBE_PAIRS` lists the pairs that meet along one.
  - They are the left oblique, the right oblique (upper and middle right against lower right) and the right horizontal fissure.
"""

import numpy as np
from jaxtyping import Bool, Float
from scipy import ndimage

LEFT_UPPER, LEFT_LOWER, RIGHT_UPPER, RIGHT_MIDDLE, RIGHT_LOWER = 10, 11, 12, 13, 14
FISSURE_LOBE_PAIRS = [
    (LEFT_UPPER, LEFT_LOWER),
    (RIGHT_UPPER, RIGHT_LOWER),
    (RIGHT_MIDDLE, RIGHT_LOWER),
    (RIGHT_UPPER, RIGHT_MIDDLE),
]

Spacing = tuple[float, float, float]


def distance_to_mask_mm(
    mask: Bool[np.ndarray, "x y z"], spacing_mm: Spacing
) -> Float[np.ndarray, "x y z"]:
    return ndimage.distance_transform_edt(~mask, sampling=spacing_mm)
