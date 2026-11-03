"""The islands of a fissure sheet: its 26-connected pieces, ranked by size.

- Rank 0 is the largest island, rank 1 the next, and so on.
  - So `sizes[rank]` is the size of the island of that rank, with no offset.
  - Background voxels have the rank `BACKGROUND`, which is -1 since 0 is an island.
- Centroids are in scanner millimetres, through the affine, so distances do not depend on the voxel spacing.
"""

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy import ndimage

from fissure_annotation.imaging.lines import CONNECT_26

BACKGROUND = -1

IslandRanks = Int[np.ndarray, " n"]
RankVolume = Int[np.ndarray, "x y z"]


def connected_pieces(mask: Bool[np.ndarray, "x y z"]) -> tuple[Int[np.ndarray, "x y z"], Int[np.ndarray, " n"]]:
    """the piece of every voxel, and the voxel count of each piece

    `ndimage.label` numbers pieces from 1 and gives background 0.
    The pieces are shifted down by one here, so this is the only place that knows about it.
    """
    labels, count = ndimage.label(mask, CONNECT_26)
    pieces = labels - 1
    return pieces, np.bincount(pieces[mask], minlength=count)


def ranks_by_size(mask: Bool[np.ndarray, "x y z"]) -> tuple[RankVolume, Int[np.ndarray, " n"]]:
    "the rank of every voxel's island, and the voxel count of each rank"
    pieces, sizes = connected_pieces(mask)
    largest_first = np.argsort(sizes, kind="stable")[::-1]
    rank_of_piece = np.empty(len(sizes), dtype=np.int32)
    rank_of_piece[largest_first] = np.arange(len(sizes))
    ranks = np.full(mask.shape, BACKGROUND, dtype=np.int32)
    ranks[mask] = rank_of_piece[pieces[mask]]
    return ranks, sizes[largest_first]


class Islands:
    def __init__(self, mask: Bool[np.ndarray, "x y z"], affine: Float[np.ndarray, "4 4"]):
        self.ranks, self.sizes = ranks_by_size(mask)
        self.affine = affine

    @property
    def count(self) -> int:
        return len(self.sizes)

    @property
    def mask(self) -> Bool[np.ndarray, "x y z"]:
        return self.ranks != BACKGROUND

    def without(self, ranks: IslandRanks) -> "Islands":
        return Islands(self.mask & ~np.isin(self.ranks, ranks), self.affine)

    def centroid_mm(self, rank: int) -> Float[np.ndarray, "3"]:
        voxel = np.array(ndimage.center_of_mass(self.ranks == rank))
        return self.affine[:3, :3] @ voxel + self.affine[:3, 3]

    def centroid_distance_mm(self, first: int, second: int) -> float:
        return float(np.linalg.norm(self.centroid_mm(first) - self.centroid_mm(second)))

    def two_largest_centroid_distance_mm(self) -> float:
        largest, second_largest = 0, 1
        return self.centroid_distance_mm(largest, second_largest)
