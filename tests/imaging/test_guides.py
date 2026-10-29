import numpy as np

from fissure_annotation.imaging.guides import draw_band_outlines, outline_of

CYAN = (0, 220, 255)


def test_the_outline_of_a_horizontal_band_is_its_top_and_bottom_rows_and_its_ends():
    region = np.zeros((20, 20), dtype=bool)
    region[8:13, 2:18] = True
    outline = outline_of(region)
    assert outline[8, 2:18].all() and outline[12, 2:18].all()
    assert not outline[10, 5:15].any()


def test_outlines_are_dashed_and_coloured_and_leave_the_inside_alone():
    picture = np.full((20, 20), 100, dtype=np.uint8)
    bands = np.zeros((20, 20), dtype=np.uint8)
    bands[8:13, 2:18] = 170
    rgb = draw_band_outlines(picture, bands, {170: CYAN})
    top = (rgb[8, 2:18] == CYAN).all(axis=-1)
    assert 0 < top.sum() < 16  # dashed, not solid
    assert (rgb[10, 5:15] == 100).all()
