from PIL import Image

from fissure_annotation.drawing.lines import (
    LINE_COLOURS,
    OWN_WAYPOINTS_COLOUR,
    draw_marked_line,
    draw_traced_line,
    line_colour,
)

RED = (255, 0, 0)


def blank() -> Image.Image:
    return Image.new("RGB", (100, 100), "black")


def test_colours_cycle_through_the_palette():
    assert line_colour(0) == line_colour(len(LINE_COLOURS))


def test_traced_line_is_coloured_and_own_waypoints_are_white():
    img = blank()
    draw_traced_line(img, own_waypoints=[(10, 20), (90, 20)], traced=[(10, 60), (90, 60)], colour=RED, label="1 R")
    assert img.getpixel((50, 60)) == RED
    assert img.getpixel((50, 20)) == OWN_WAYPOINTS_COLOUR


def test_marked_line_circles_each_point():
    img = blank()
    draw_marked_line(img, [(20, 50), (80, 50)], RED, "1 L")
    assert img.getpixel((50, 50)) == RED
    assert img.getpixel((20, 53)) == RED  # bottom of the circle round the first point


def test_label_sits_beside_the_line_start():
    img = blank()
    draw_marked_line(img, [(20, 80), (80, 80)], RED, "7 R")
    assert any(img.getpixel((x, y)) != (0, 0, 0) for x in range(25, 40) for y in range(64, 78))  # antialiased, so not exactly RED
