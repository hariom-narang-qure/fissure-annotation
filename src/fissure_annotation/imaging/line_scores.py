"""Score the traced lines of one slice against the fissures of a manual lobe segmentation.

Why:
- Blind reviews of overlays were too noisy to tell small changes to the agent apart.
- A manual lobe segmentation (LOCCA) gives the fissures as the borders between lobes, so a slice can be scored without a reviewer.

How:
1. The reference line is every lobe pixel that touches the other lobe of a fissure pair (lobes.FISSURE_LOBE_PAIRS).
   - Touching means 8-connected, so a diagonal border is still one pixel wide on each side.
2. Distances are in millimetres, from the pixel spacing of the slice, since coronal and sagittal pixels need not be square.
3. Precision: the share of traced pixels within `tolerance_mm` of the reference line.
4. Recall: the share of reference pixels within `tolerance_mm` of a traced pixel.

Caveats:
- A lobe border exists where a fissure is incomplete or invisible, so recall also counts lines no reader could see.
- Manual borders are a few millimetres off in places, which is what the tolerance absorbs.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Integer
from scipy import ndimage

from fissure_annotation.imaging.lobes import FISSURE_LOBE_PAIRS

DEFAULT_TOLERANCE_MM = 3.0
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)

PixelSpacing = tuple[float, float]  # rows, columns in mm


class LineScore(NamedTuple):
    traced_px: int
    traced_near_reference_px: int
    reference_px: int
    reference_near_traced_px: int


def touching(first: Bool[np.ndarray, "h w"], second: Bool[np.ndarray, "h w"]) -> Bool[np.ndarray, "h w"]:
    return first & ndimage.binary_dilation(second, EIGHT_CONNECTED)


def reference_fissure_lines(lobes: Integer[np.ndarray, "h w"]) -> Bool[np.ndarray, "h w"]:
    lines = np.zeros(lobes.shape, dtype=bool)
    for lobe_a, lobe_b in FISSURE_LOBE_PAIRS:
        in_a, in_b = lobes == lobe_a, lobes == lobe_b
        lines |= touching(in_a, in_b) | touching(in_b, in_a)
    return lines


def near(mask: Bool[np.ndarray, "h w"], spacing: PixelSpacing, tolerance_mm: float) -> Bool[np.ndarray, "h w"]:
    # The distance transform of an empty mask is all zeros, which would call everything near.
    if not mask.any():
        return np.zeros_like(mask)
    return ndimage.distance_transform_edt(~mask, sampling=spacing) <= tolerance_mm


def score_lines(
    traced: Bool[np.ndarray, "h w"],
    lobes: Integer[np.ndarray, "h w"],
    spacing: PixelSpacing,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> LineScore:
    reference = reference_fissure_lines(lobes)
    return LineScore(
        traced_px=int(traced.sum()),
        traced_near_reference_px=int((traced & near(reference, spacing, tolerance_mm)).sum()),
        reference_px=int(reference.sum()),
        reference_near_traced_px=int((reference & near(traced, spacing, tolerance_mm)).sum()),
    )
