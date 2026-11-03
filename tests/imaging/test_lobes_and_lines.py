import numpy as np

from fissure_annotation.imaging.lines import keep_lines_mostly_inside, keep_lines_near
from fissure_annotation.imaging.lobes import distance_to_mask_mm

ISOTROPIC = (1.0, 1.0, 1.0)


def test_distance_to_mask_uses_the_spacing():
    mask = np.zeros((10, 3, 3), dtype=bool)
    mask[0] = True
    assert distance_to_mask_mm(mask, (2.0, 1.0, 1.0))[3, 1, 1] == 6.0


def volume_with_reference_at_x0():
    reference = np.zeros((60, 6, 6), dtype=bool)
    reference[0] = True
    return distance_to_mask_mm(reference, ISOTROPIC)


def test_a_line_with_a_point_near_the_reference_is_kept_and_a_far_line_is_removed():
    distance = volume_with_reference_at_x0()
    mask = np.zeros(distance.shape, dtype=bool)
    mask[1:10, 2, 2] = True
    mask[40:50, 4, 4] = True
    kept, removed = keep_lines_near(mask, distance, max_distance_mm=5.0)
    assert kept[1:10, 2, 2].all() and not kept[40:50, 4, 4].any()
    assert removed[40:50, 4, 4].all() and not removed[1:10, 2, 2].any()


def test_a_long_line_with_only_its_start_near_the_reference_is_kept_whole():
    "a real fissure with a stray tail must not be removed because of the tail"
    distance = volume_with_reference_at_x0()
    mask = np.zeros(distance.shape, dtype=bool)
    mask[2:55, 2, 2] = True
    kept, removed = keep_lines_near(mask, distance, max_distance_mm=5.0)
    assert (kept == mask).all() and not removed.any()


def test_voxels_touching_only_at_a_corner_are_one_line():
    distance = volume_with_reference_at_x0()
    mask = np.zeros(distance.shape, dtype=bool)
    mask[3, 2, 2] = True
    mask[4, 3, 3] = True
    mask[5, 4, 4] = True
    mask[30, 2, 2] = True  # a separate line, far away
    kept, _ = keep_lines_near(mask, distance, max_distance_mm=4.0)
    assert kept[3:6].sum() == 3 and not kept[30].any()


def test_kept_and_removed_split_the_mask_without_overlap():
    distance = volume_with_reference_at_x0()
    mask = np.zeros(distance.shape, dtype=bool)
    mask[1:5, 2, 2] = True
    mask[40:45, 4, 4] = True
    kept, removed = keep_lines_near(mask, distance, max_distance_mm=3.0)
    assert (kept | removed == mask).all() and not (kept & removed).any()


def test_a_line_mostly_inside_the_region_is_kept_and_a_line_outside_is_removed():
    region = np.zeros((40, 12, 12), dtype=bool)
    region[:20] = True
    mask = np.zeros(region.shape, dtype=bool)
    mask[2:18, 1, 1] = True  # inside
    mask[25:38, 5, 5] = True  # outside
    mask[10:30, 9, 9] = True  # exactly half of its voxels are inside
    kept, removed = keep_lines_mostly_inside(mask, region, min_share=0.5)
    assert kept[2:18, 1, 1].all() and not kept[25:38, 5, 5].any()
    assert kept[10:30, 9, 9].all() and removed[25:38, 5, 5].all()


def test_a_line_with_a_short_tail_outside_the_region_is_kept_whole():
    region = np.zeros((60, 6, 6), dtype=bool)
    region[:40] = True
    mask = np.zeros(region.shape, dtype=bool)
    mask[2:50, 2, 2] = True  # 38 of 48 voxels inside, the tail goes beyond the region
    kept, removed = keep_lines_mostly_inside(mask, region, min_share=0.5)
    assert (kept == mask).all() and not removed.any()
