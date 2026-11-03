"""Score the traced lines of one slice against reference lines on the same slice.

Why:
- Blind reviews of overlays were too noisy to tell small changes to the agent apart.
- Reference lines drawn on the slice, such as reviewed agent traces, let a slice be scored without a reviewer.
  - An earlier eval used the borders of manual lobes instead. A border exists where no fissure is visible, so recall had a low ceiling.

How:
1. Distances are in millimetres, from the pixel spacing of the slice, since coronal and sagittal pixels need not be square.
2. Precision: the share of traced pixels within `tolerance_mm` of a reference pixel.
3. Recall: the share of reference pixels within `tolerance_mm` of a traced pixel.
4. The counts are kept, not the ratios, so slices can be pooled by pixels.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool
from scipy import ndimage

DEFAULT_TOLERANCE_MM = 3.0

PixelSpacing = tuple[float, float]  # rows, columns in mm


class LineScore(NamedTuple):
    traced_px: int
    traced_near_reference_px: int
    reference_px: int
    reference_near_traced_px: int


def near(mask: Bool[np.ndarray, "h w"], spacing: PixelSpacing, tolerance_mm: float) -> Bool[np.ndarray, "h w"]:
    # The distance transform of an empty mask is all zeros, which would call everything near.
    if not mask.any():
        return np.zeros_like(mask)
    return ndimage.distance_transform_edt(~mask, sampling=spacing) <= tolerance_mm


def score_lines(
    traced: Bool[np.ndarray, "h w"],
    reference: Bool[np.ndarray, "h w"],
    spacing: PixelSpacing,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> LineScore:
    return LineScore(
        traced_px=int(traced.sum()),
        traced_near_reference_px=int((traced & near(reference, spacing, tolerance_mm)).sum()),
        reference_px=int(reference.sum()),
        reference_near_traced_px=int((reference & near(traced, spacing, tolerance_mm)).sum()),
    )
