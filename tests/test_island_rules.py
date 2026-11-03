import numpy as np

from fissure_annotation.island_rules import DropSmallIslands, apply_island_rules
from fissure_annotation.imaging.islands import Islands

SIZE = 40


def sheet_of_blocks(*sizes_and_x):
    "one flat block of n voxels per (n, x), each at its own x so they never touch"
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    for voxels, x in sizes_and_x:
        rows, rest = divmod(voxels, SIZE)
        mask[x, :rows, :] = True
        mask[x, rows, :rest] = True
    return mask


def test_islands_under_the_share_of_the_largest_are_dropped():
    islands = Islands(sheet_of_blocks((1000, 0), (600, 10), (49, 20), (50, 30)), np.eye(4))
    kept = DropSmallIslands(min_share_of_largest=0.05).apply(islands).kept
    assert kept.sizes.tolist() == [1000, 600, 50]


def test_dropping_from_an_empty_sheet_keeps_it_empty():
    islands = Islands(np.zeros((SIZE, SIZE, SIZE), dtype=bool), np.eye(4))
    assert DropSmallIslands().apply(islands).kept.count == 0


def test_the_report_has_the_sizes_before_and_after_and_the_removed_mask():
    sheet = sheet_of_blocks((1000, 0), (600, 10), (20, 20))
    trimmed = apply_island_rules([DropSmallIslands()], sheet, np.eye(4))
    assert trimmed.report.island_voxels_before == [1000, 600, 20]
    assert trimmed.report.island_voxels_after == [1000, 600]
    assert [removal.rule for removal in trimmed.removals] == ["drop_small_islands"]
    assert (trimmed.removals[0].removed == sheet_of_blocks((20, 20))).all()
    assert not trimmed.report.needs_review


def test_no_rules_keep_the_sheet_as_it_is():
    sheet = sheet_of_blocks((1000, 0), (20, 20))
    trimmed = apply_island_rules([], sheet, np.eye(4))
    assert (trimmed.kept == sheet).all() and trimmed.removals == []
