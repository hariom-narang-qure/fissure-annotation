import numpy as np

from fissure_annotation.imaging.vessels import component_elongations, draw_dots, vessel_cores, vessel_dots_along_skeleton

BLUE = (40, 90, 255)


def disc(shape, centre, radius):
    rows, columns = np.indices(shape)
    return (rows - centre[1]) ** 2 + (columns - centre[0]) ** 2 <= radius**2


def test_each_component_gets_one_core_at_its_deepest_pixel():
    mask = disc((64, 64), (20, 30), 5) | disc((64, 64), (45, 10), 3)
    assert sorted(vessel_cores(mask, min_area_px=4)) == [(20, 30), (45, 10)]


def test_components_under_the_minimum_area_get_no_core():
    mask = disc((64, 64), (20, 30), 5)
    mask[50:52, 50] = True  # 2 px of noise
    assert vessel_cores(mask, min_area_px=4) == [(20, 30)]


def test_the_core_of_a_bent_component_lies_inside_it_where_its_centroid_would_not():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:10] = True  # the long arm of an L
    mask[30:35, 5:35] = True  # its foot
    (x, y), = vessel_cores(mask, min_area_px=4)
    assert mask[y, x]


def test_an_empty_mask_has_no_cores():
    assert vessel_cores(np.zeros((8, 8), dtype=bool), min_area_px=1) == []


def test_a_disc_is_round_and_a_line_is_elongated():
    mask = disc((64, 64), (15, 15), 5)
    mask[50:53, 5:60] = True
    round_one, line = component_elongations(mask, min_area_px=4)
    assert round_one < 1.2 and line > 10


def test_dots_along_the_skeleton_keep_a_disc_to_one_dot_and_spread_along_a_line():
    mask = disc((64, 64), (15, 15), 5)
    mask[50:53, 4:60] = True
    dots = vessel_dots_along_skeleton(mask, min_area_px=4, spacing_px=10)
    on_disc = [(x, y) for x, y in dots if y < 30]
    on_line = sorted(x for x, y in dots if y >= 30)
    assert on_disc == [(15, 15)]
    assert all(50 <= y <= 52 for x, y in dots if y >= 30)
    assert on_line[0] <= 9 and on_line[-1] >= 54  # both ends are within half the spacing of a dot
    assert all(4 <= b - a <= 10 for a, b in zip(on_line, on_line[1:]))


def test_dots_are_discs_of_one_colour_on_an_rgb_copy():
    picture = np.full((20, 20), 100, dtype=np.uint8)
    rgb = draw_dots(picture, [(10, 10), (0, 0)], BLUE, radius_px=2)
    blue = (rgb == BLUE).all(axis=-1)
    assert blue[10, 8:13].all() and blue[8:13, 10].all()
    assert not blue[8, 8]  # a corner of the box is outside the disc
    assert blue[0, 0] and blue[2, 0]  # a dot at the edge is clipped, not refused
    assert (rgb[15, 15] == 100).all() and (picture == 100).all()


def test_dots_keep_what_is_already_drawn_on_an_rgb_picture():
    picture = np.zeros((20, 20, 3), dtype=np.uint8)
    picture[0, :] = (255, 220, 0)
    rgb = draw_dots(picture, [(10, 10)], BLUE, radius_px=2)
    assert (rgb[0, :] == (255, 220, 0)).all() and (rgb[10, 10] == BLUE).all()
