"""The islands of a fissure sheet: its 26-connected pieces, numbered by size.

- Island 1 is the largest, island 2 the next, and so on.
  - Rules that look at "the two main islands" then read the first two numbers.
- Centroids are in scanner millimetres, through the affine, so distances do not depend on the voxel spacing.
"""

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy import ndimage

from fissure_annotation.imaging.lines import CONNECT_26

IslandNumbers = Int[np.ndarray, " n"]


def labels_numbered_by_size(
    mask: Bool[np.ndarray, "x y z"],
) -> tuple[Int[np.ndarray, "x y z"], Int[np.ndarray, " n"]]:
    "the island number of every voxel (0 is background), and the voxel count of each island, largest first"
    labels, count = ndimage.label(mask, CONNECT_26)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    largest_first = np.argsort(sizes, kind="stable")[::-1]
    new_number = np.zeros(count + 1, dtype=np.int32)
    new_number[largest_first + 1] = np.arange(1, count + 1)
    return new_number[labels], sizes[largest_first]


class Islands:
    "the islands of one sheet, with island i+1 having sizes[i] voxels"

    def __init__(self, mask: Bool[np.ndarray, "x y z"], affine: Float[np.ndarray, "4 4"]):
        self.labels, self.sizes = labels_numbered_by_size(mask)
        self.affine = affine

    @property
    def count(self) -> int:
        return len(self.sizes)

    @property
    def mask(self) -> Bool[np.ndarray, "x y z"]:
        return self.labels > 0

    def without(self, numbers: IslandNumbers) -> "Islands":
        return Islands(self.mask & ~np.isin(self.labels, numbers), self.affine)

    def centroid_mm(self, number: int) -> Float[np.ndarray, "3"]:
        voxel = np.array(ndimage.center_of_mass(self.labels == number))
        return self.affine[:3, :3] @ voxel + self.affine[:3, 3]

    def centroid_distance_mm(self, first: int, second: int) -> float:
        return float(np.linalg.norm(self.centroid_mm(first) - self.centroid_mm(second)))
