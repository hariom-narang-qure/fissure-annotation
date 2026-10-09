"""Pick slices where the lobe masks say fissures cross: those with the most lobe-lobe boundary pixels.

Two per plane: the best axial in each half of the lung's z range, the best coronal in each half of its y range.
The mask is imperfect, so this is only a way to avoid slices with no fissure at all.
"""
import numpy as np

from fissure_annotation.slicing import Case, boundary_mask, cut


def boundary_length(case: Case, plane: str, index: int) -> int:
    return int(boundary_mask(cut(case, plane, index).lobes).sum())


def best_in(case: Case, plane: str, lo: int, hi: int, step: int = 8) -> int:
    idx = list(range(lo, hi, step))
    return idx[int(np.argmax([boundary_length(case, plane, i) for i in idx]))]


def pick_slices(case: Case) -> list[tuple[str, int]]:
    picks = []
    for plane, axis in [("axial", 2), ("coronal", 1)]:
        present = np.nonzero((case.lobes > 0).sum(axis=tuple(i for i in range(3) if i != axis)) > 500)[0]
        lo, hi = int(present[0]), int(present[-1])
        mid = (lo + hi) // 2
        picks += [(plane, best_in(case, plane, lo, mid)), (plane, best_in(case, plane, mid, hi))]
    return picks
