"""Where fissures should be, from a lung lobe segmentation.

Why:
- A fissure is the interface between two lobes, so the lobe interfaces say roughly where fissures are.
- The segmentation is slightly off near the fissures, so this is a rough guide for checking lines, never the answer.

How:
- For each pair of lobes that meet along a fissure, keep the voxels of either lobe within `band_mm` of the other lobe.
  - Distances use the voxel spacing, so the band is in millimetres.
  - Lobes do not always touch, the segmentation can leave unlabelled voxels between them.
    - A distance threshold still finds the boundary across a gap that is no wider than the band.
    - Only lobe voxels are kept, never the gap voxels.
- A lobe missing from the segmentation gives no band for its pairs.
- The pairs are the left oblique, the right oblique (upper and middle right against lower right) and the right horizontal fissure.
- Lobe ids are the TotalSegmentator ids of the `total` task.
"""

import numpy as np
from jaxtyping import Bool, Float, Integer
from scipy import ndimage

LEFT_UPPER, LEFT_LOWER, RIGHT_UPPER, RIGHT_MIDDLE, RIGHT_LOWER = 10, 11, 12, 13, 14
FISSURE_LOBE_PAIRS = [
    (LEFT_UPPER, LEFT_LOWER),
    (RIGHT_UPPER, RIGHT_LOWER),
    (RIGHT_MIDDLE, RIGHT_LOWER),
    (RIGHT_UPPER, RIGHT_MIDDLE),
]
DEFAULT_BAND_MM = 2.0

Spacing = tuple[float, float, float]


def distance_to_mask_mm(
    mask: Bool[np.ndarray, "x y z"], spacing_mm: Spacing
) -> Float[np.ndarray, "x y z"]:
    return ndimage.distance_transform_edt(~mask, sampling=spacing_mm)


def voxels_of_first_near_second(
    first: Bool[np.ndarray, "x y z"],
    second: Bool[np.ndarray, "x y z"],
    spacing_mm: Spacing,
    band_mm: float,
) -> Bool[np.ndarray, "x y z"]:
    # The distance transform of an empty mask is all zeros, which would label all of `first`.
    if not second.any():
        return np.zeros_like(first)
    return first & (distance_to_mask_mm(second, spacing_mm) <= band_mm)


def interface_between(
    lobes: Integer[np.ndarray, "x y z"],
    lobe_a: int,
    lobe_b: int,
    spacing_mm: Spacing,
    band_mm: float,
) -> Bool[np.ndarray, "x y z"]:
    in_a, in_b = lobes == lobe_a, lobes == lobe_b
    return voxels_of_first_near_second(
        in_a, in_b, spacing_mm, band_mm
    ) | voxels_of_first_near_second(in_b, in_a, spacing_mm, band_mm)


def lobe_boundary_band(
    lobes: Integer[np.ndarray, "x y z"],
    spacing_mm: Spacing,
    band_mm: float = DEFAULT_BAND_MM,
) -> Bool[np.ndarray, "x y z"]:
    band = np.zeros(lobes.shape, dtype=bool)
    for lobe_a, lobe_b in FISSURE_LOBE_PAIRS:
        band |= interface_between(lobes, lobe_a, lobe_b, spacing_mm, band_mm)
    return band
