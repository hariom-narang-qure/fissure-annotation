"""Drop the lines of a mask that have nothing near a reference.

Why:
- An agent line with no point near the expected lobe boundary is most likely not a fissure trace.
  - Examples are the outline of the body, the table, the diaphragm and the hilum.
- The lobe segmentation is slightly off near the fissures, so "near" is a wide distance.
- A line is judged by its NEAREST point, not its farthest.
  - A real fissure is one long line, and a short tail that wanders off would remove all of it.
  - On a real scan, judging by the farthest point removed 70% of the axial mask and 90% of the sagittal one.
  - Those were long lines that lie on the lobe boundary, with a short stretch beyond the distance.

How:
- A line is a 26-connected component of the mask.
- The distance of a line is the smallest distance from any of its voxels to the reference, in millimetres.
- A line within `max_distance_mm` is kept whole, any other line is removed whole.

`keep_lines_mostly_inside` is the rule that does not depend on where the lobe interfaces are.
- It judges a line by the share of its voxels inside a region, such as the lungs with a margin.
- A line that is mostly outside the lungs is the outline of the body or the table, which is the part of the nearest point rule that matters most.
- It does not need the lobe segmentation to be right inside the lungs, which it is not in severe emphysema.

`keep_large_pieces` is the plain size rule for what other filters leave behind, a few specks or short stubs.
"""

import numpy as np
from jaxtyping import Bool, Float
from scipy import ndimage

CONNECT_26 = np.ones((3, 3, 3), dtype=bool)


def nearest_distance_per_line(
    mask: Bool[np.ndarray, "x y z"], distance_mm: Float[np.ndarray, "x y z"]
) -> tuple[np.ndarray, np.ndarray]:
    "the line number of every voxel (0 is not a line), and the nearest distance of each line, lines numbered from 1"
    lines, count = ndimage.label(mask, CONNECT_26)
    nearest = ndimage.minimum(distance_mm, lines, index=np.arange(1, count + 1))
    return lines, np.asarray(nearest)


def keep_lines_near(
    mask: Bool[np.ndarray, "x y z"],
    distance_mm: Float[np.ndarray, "x y z"],
    max_distance_mm: float,
) -> tuple[Bool[np.ndarray, "x y z"], Bool[np.ndarray, "x y z"]]:
    "the lines that have a point within the distance, and the lines that do not"
    lines, nearest = nearest_distance_per_line(mask, distance_mm)
    kept_line_numbers = np.flatnonzero(nearest <= max_distance_mm) + 1
    kept = np.isin(lines, kept_line_numbers)
    return kept, mask & ~kept


def keep_large_pieces(
    mask: Bool[np.ndarray, "x y z"], min_voxels: int
) -> tuple[Bool[np.ndarray, "x y z"], Bool[np.ndarray, "x y z"]]:
    "the 26-connected pieces of at least min_voxels voxels, and the smaller ones"
    pieces, count = ndimage.label(mask, CONNECT_26)
    sizes = np.bincount(pieces.ravel(), minlength=count + 1)
    kept = np.isin(pieces, np.flatnonzero(sizes[1:] >= min_voxels) + 1)
    return kept, mask & ~kept


def keep_lines_mostly_inside(
    mask: Bool[np.ndarray, "x y z"],
    region: Bool[np.ndarray, "x y z"],
    min_share: float,
) -> tuple[Bool[np.ndarray, "x y z"], Bool[np.ndarray, "x y z"]]:
    "the lines with at least min_share of their voxels inside the region, and the other lines"
    lines, count = ndimage.label(mask, CONNECT_26)
    share = ndimage.mean(region, lines, index=np.arange(1, count + 1))
    kept = np.isin(lines, np.flatnonzero(np.asarray(share) >= min_share) + 1)
    return kept, mask & ~kept
