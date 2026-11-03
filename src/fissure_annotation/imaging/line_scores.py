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
   - A slice with long fissures then weighs more than one with a short stub.

What the scores mean:
- A trace up to `tolerance_mm` off the reference counts as right. That absorbs the small offset between two traces of one fissure.
- A short trace on the right fissure has full precision and partial recall.
- A trace on a vessel, far from any fissure, lowers precision and leaves recall as it is.
- A slice whose reference is empty, with no visible fissure, adds nothing to recall, and any trace on it counts against precision.
- Neither score knows which fissure is which. A traced line near any reference line counts.
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
    """every pixel within tolerance_mm of the mask: a band about twice the tolerance wide, around its lines

    How:
    1. `~mask` flips the mask, so line pixels are False and every other pixel is True.
    2. `distance_transform_edt` gives each True pixel its straight-line distance to the nearest False pixel.
       - After the flip that is each pixel's distance to the nearest line pixel. Line pixels get 0.
    3. `sampling=spacing` measures it in millimetres, one row step and one column step each by their own size.
    4. `<= tolerance_mm` keeps the pixels close enough.

    For example, with pixels 0.7 mm apart and a line at column 5:

        column:       0    1    2    3    4    5    6    7    8    9
        mask:         .    .    .    .    .    #    .    .    .    .
        distance mm:  3.5  2.8  2.1  1.4  0.7  0    0.7  1.4  2.1  2.8
        near at 3 mm: no   yes  yes  yes  yes  yes  yes  yes  yes  yes

    score_lines counts the pixels of one set that fall in the band of the other.
    """
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
    """the precision side counts traced pixels in the band around the reference, the recall side reference pixels in the band around the trace

    A traced pixel outside the reference's band is a false trace. A reference pixel outside the trace's band was missed.
    """
    return LineScore(
        traced_px=int(traced.sum()),
        traced_near_reference_px=int((traced & near(reference, spacing, tolerance_mm)).sum()),
        reference_px=int(reference.sum()),
        reference_near_traced_px=int((reference & near(traced, spacing, tolerance_mm)).sum()),
    )
