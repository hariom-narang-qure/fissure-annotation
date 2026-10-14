import numpy as np

from fissure_annotation.imaging.lines import keep_lines_near
from fissure_annotation.imaging.lobes import (
    LEFT_LOWER,
    LEFT_UPPER,
    RIGHT_UPPER,
    distance_to_mask_mm,
    lobe_boundary_band,
)

ISOTROPIC = (1.0, 1.0, 1.0)


def two_lobes_side_by_side(upper=LEFT_UPPER, lower=LEFT_LOWER, gap=0):
    "lobe `upper` for x < 10, then `gap` unlabelled layers, then lobe `lower`"
    lobes = np.zeros((20, 4, 4), dtype=np.uint8)
    lobes[:10] = upper
    lobes[10 + gap :] = lower
    return lobes


def band_x_positions(band):
    return sorted(set(np.nonzero(band)[0].tolist()))


def test_the_band_has_the_voxels_of_both_lobes_within_the_band_of_the_other():
    band = lobe_boundary_band(two_lobes_side_by_side(), ISOTROPIC, band_mm=2.0)
    assert band_x_positions(band) == [8, 9, 10, 11]


def test_the_band_is_measured_in_millimetres_not_voxels():
    band = lobe_boundary_band(two_lobes_side_by_side(), (2.0, 1.0, 1.0), band_mm=2.0)
    assert band_x_positions(band) == [9, 10]


def test_a_gap_no_wider_than_the_band_is_crossed_but_never_labelled():
    band = lobe_boundary_band(two_lobes_side_by_side(gap=1), ISOTROPIC, band_mm=2.0)
    assert band_x_positions(band) == [9, 11]  # x=10 is the gap


def test_lobes_that_do_not_share_a_fissure_give_no_band():
    band = lobe_boundary_band(
        two_lobes_side_by_side(upper=LEFT_UPPER, lower=RIGHT_UPPER),
        ISOTROPIC,
        band_mm=2.0,
    )
    assert not band.any()


def test_a_lobe_missing_from_the_segmentation_gives_no_band_for_its_pairs():
    lobes = np.full((20, 4, 4), LEFT_UPPER, dtype=np.uint8)  # no lower left lobe
    assert not lobe_boundary_band(lobes, ISOTROPIC, band_mm=2.0).any()


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
