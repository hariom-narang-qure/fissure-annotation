import pytest

from fissure_annotation.frames import CropFrame, main_to_view, view_to_main

CROP = CropFrame((100, 50, 300, 250))


def test_crop_corners_map_to_the_main_frame_corners():
    assert CROP.to_main(100, 50) == (0, 0)
    assert CROP.to_main(300, 250) == (1000, 1000)


def test_main_frame_to_pixels_and_back():
    for x, y in [(0, 0), (250, 500), (1000, 1000)]:
        col, row = CROP.to_pixels(x, y)
        assert CROP.to_main(col, row) == (x, y)


def test_view_frame_is_a_box_inside_the_main_frame():
    zoom = (200.0, 300.0, 600.0, 700.0)
    assert view_to_main(0, 0, zoom) == (200.0, 300.0)
    assert view_to_main(1000, 1000, zoom) == (600.0, 700.0)
    assert main_to_view(300.0, 600.0, zoom) == (250.0, 750.0)


def test_a_zoom_of_a_zoom_lands_in_the_right_main_frame_place():
    first = (200.0, 200.0, 600.0, 600.0)
    inner_in_first = (500, 500, 1000, 1000)  # lower right quarter of the first zoom, in its own frame
    x0, y0 = view_to_main(inner_in_first[0], inner_in_first[1], first)
    x1, y1 = view_to_main(inner_in_first[2], inner_in_first[3], first)
    second = (x0, y0, x1, y1)
    assert second == (400.0, 400.0, 600.0, 600.0)
    assert view_to_main(500, 500, second) == (500.0, 500.0)


def test_frames_are_immutable():
    with pytest.raises(AttributeError):
        CROP.box = (0, 0, 1, 1)
