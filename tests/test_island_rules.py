import numpy as np
from pydantic import BaseModel

from fissure_annotation.island_rules import (
    DropIslandsUnder,
    DropSmallIslands,
    FlagCloseMainIslands,
    FlagExtraIslands,
    FlagTooFewIslands,
    IslandRule,
    apply_island_rules,
    default_island_rules,
)
from fissure_annotation.imaging.islands import Islands
from fissure_annotation.removals import RemovedVoxels

SIZE = 40


class PostProcessRules(BaseModel):
    rules: list[IslandRule]


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
    removed = RemovedVoxels(sheet.shape)
    trimmed = apply_island_rules([DropSmallIslands()], sheet, np.eye(4), removed)
    assert trimmed.report.island_voxels_before == [1000, 600, 20]
    assert trimmed.report.island_voxels_after == [1000, 600]
    assert [step.step for step in removed.steps] == ["drop_small_islands"]
    assert ((removed.labels == 1) == sheet_of_blocks((20, 20))).all()
    assert not trimmed.report.needs_review


def test_no_rules_keep_the_sheet_as_it_is():
    sheet = sheet_of_blocks((1000, 0), (20, 20))
    removed = RemovedVoxels(sheet.shape)
    trimmed = apply_island_rules([], sheet, np.eye(4), removed)
    assert (trimmed.kept == sheet).all() and removed.steps == []


def test_more_than_two_islands_are_flagged_and_kept():
    islands = Islands(sheet_of_blocks((1000, 0), (600, 10), (300, 20)), np.eye(4))
    outcome = FlagExtraIslands().apply(islands)
    assert outcome.kept.count == 3
    assert [flag.rule for flag in outcome.flags] == ["flag_extra_islands"]


def test_two_islands_are_not_flagged():
    islands = Islands(sheet_of_blocks((1000, 0), (600, 10)), np.eye(4))
    assert FlagExtraIslands().apply(islands).flags == []


def test_islands_dropped_as_small_are_not_counted_as_extra():
    sheet = sheet_of_blocks((1000, 0), (600, 10), (20, 20))
    trimmed = apply_island_rules([DropSmallIslands(), FlagExtraIslands()], sheet, np.eye(4), RemovedVoxels(sheet.shape))
    assert not trimmed.report.needs_review


def test_a_config_can_list_the_rules_by_kind():
    rules = PostProcessRules.model_validate(
        {"rules": [{"kind": "drop_small_islands", "min_share_of_largest": 0.1}, {"kind": "flag_extra_islands"}]}
    ).rules
    assert rules == [DropSmallIslands(min_share_of_largest=0.1), FlagExtraIslands()]
    assert default_island_rules() == [
        DropIslandsUnder(),
        DropSmallIslands(),
        FlagExtraIslands(max_islands=2),
        FlagTooFewIslands(),
        FlagCloseMainIslands(),
    ]


def test_islands_under_the_voxel_floor_are_dropped():
    islands = Islands(sheet_of_blocks((1000, 0), (150, 10), (149, 20)), np.eye(4))
    assert DropIslandsUnder(min_voxels=150).apply(islands).kept.sizes.tolist() == [1000, 150]


def test_a_sheet_of_only_specks_keeps_nothing():
    islands = Islands(sheet_of_blocks((100, 0), (90, 10)), np.eye(4))
    rules = [DropIslandsUnder(min_voxels=150), DropSmallIslands()]
    assert apply_island_rules(rules, islands.mask, np.eye(4), RemovedVoxels(islands.mask.shape)).kept.sum() == 0



def two_islands_mm_apart(distance_mm):
    "two single-voxel islands along x on a 1 mm grid"
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    mask[0, 0, 0] = mask[0, 0, 2] = True
    return Islands(mask, np.diag([1.0, 1.0, distance_mm / 2, 1.0]))


def test_main_islands_closer_than_the_distance_are_flagged():
    outcome = FlagCloseMainIslands(min_centroid_distance_mm=80).apply(two_islands_mm_apart(60))
    assert [flag.rule for flag in outcome.flags] == ["flag_close_main_islands"]
    assert "60 mm" in outcome.flags[0].reason


def test_main_islands_far_apart_are_not_flagged():
    assert FlagCloseMainIslands(min_centroid_distance_mm=80).apply(two_islands_mm_apart(150)).flags == []


def single_island():
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    mask[0, 0, 0] = True
    return Islands(mask, np.eye(4))


def test_a_single_island_is_flagged_as_too_few():
    outcome = FlagTooFewIslands().apply(single_island())
    assert [flag.rule for flag in outcome.flags] == ["flag_too_few_islands"]
    assert "islands left: 1," in outcome.flags[0].reason


def test_two_islands_are_not_too_few():
    assert FlagTooFewIslands().apply(two_islands_mm_apart(150)).flags == []


def test_a_single_island_has_no_distance_to_flag():
    assert FlagCloseMainIslands().apply(single_island()).flags == []


def test_the_distance_is_between_the_two_largest_islands():
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    mask[0, 0, :3] = True  # the largest, centred at z=1
    mask[0, 0, 10:12] = True  # the second largest, centred at z=10.5
    mask[0, 0, 30] = True  # the smallest, far from both
    assert Islands(mask, np.eye(4)).two_largest_centroid_distance_mm() == 9.5
