from PIL import Image

from fissure_annotation.drawing.compose import tile_row
from fissure_annotation.drawing.rings import RING_COLOUR, draw_point_rings

YELLOW = (255, 255, 0)


def blank(w: int = 100, h: int = 60) -> Image.Image:
    return Image.new("RGB", (w, h), "black")


def test_ring_is_hollow_so_the_line_underneath_stays_visible():
    img = blank()
    draw_point_rings(img, [(50, 30)])
    assert img.getpixel((55, 30)) == RING_COLOUR
    assert img.getpixel((50, 30)) == (0, 0, 0)


def test_points_are_joined_only_when_asked():
    plain, joined = blank(), blank()
    draw_point_rings(plain, [(10, 30), (90, 30)])
    draw_point_rings(joined, [(10, 30), (90, 30)], join_colour=YELLOW)
    assert plain.getpixel((50, 30)) == (0, 0, 0)
    assert joined.getpixel((50, 30)) == YELLOW


def test_tiles_are_laid_out_left_to_right_with_a_gap():
    left, right = Image.new("RGB", (10, 5), "red"), Image.new("RGB", (10, 5), "blue")
    out = tile_row([left, right], gap=4)
    assert out.size == (24, 5)
    assert out.getpixel((9, 0)) == (255, 0, 0)
    assert out.getpixel((11, 0)) == (0, 0, 0)
    assert out.getpixel((14, 0)) == (0, 0, 255)
