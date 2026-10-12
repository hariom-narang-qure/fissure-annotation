import numpy as np

from fissure_annotation.imaging.overlay import RED, contact_sheet, labelled, red_overlay
from fissure_annotation.imaging.ticks import MARGIN_PX, crop_with_ticks, tick_positions


def test_ticks_fall_on_multiples_of_twenty_inside_the_box():
    assert list(tick_positions(35, 100)) == [40, 60, 80]


def test_the_crop_is_scaled_and_gets_a_margin():
    image = np.zeros((100, 120), dtype=np.uint8)
    assert crop_with_ticks(image, (10, 20, 50, 70), 3).size == (40 * 3 + MARGIN_PX, 50 * 3 + MARGIN_PX)


def test_the_overlay_marks_the_mask_in_red_and_leaves_the_rest_grey():
    image = np.full((9, 9), 100, dtype=np.uint8)
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    overlay = red_overlay(image, mask)
    assert overlay[4, 4].tolist() == list(RED)
    assert overlay[0, 0].tolist() == [100, 100, 100]


def test_the_contact_sheet_pads_the_last_row():
    tiles = [np.ones((4, 5, 3), dtype=np.uint8)] * 3
    assert contact_sheet(tiles, columns=2).shape == (8, 10, 3)


def test_a_label_draws_on_the_tile_and_keeps_the_shape():
    tile = np.zeros((20, 60, 3), dtype=np.uint8)
    out = labelled(tile, "z=160")
    assert out.shape == tile.shape and out.any()
