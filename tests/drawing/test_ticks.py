from PIL import Image

from fissure_annotation.drawing.ticks import TICK_COLOUR, draw_ticks


def ticked(w: int = 400, h: int = 300) -> Image.Image:
    img = Image.new("RGB", (w, h), "black")
    draw_ticks(img)
    return img


def test_corner_ticks_sit_on_the_image_edges_so_the_frame_is_0_to_1000():
    img = ticked()
    assert img.getpixel((0, 0)) == TICK_COLOUR
    assert img.getpixel((399, 299)) == TICK_COLOUR


def test_a_tick_every_hundred_units_on_the_top_edge():
    img = ticked(w=1001)
    on_top_edge = [x for x in range(1001) if img.getpixel((x, 3)) == TICK_COLOUR]
    assert on_top_edge == list(range(0, 1001, 100))


def test_ticks_stay_inside_the_image_not_in_a_margin():
    assert ticked().size == (400, 300)


def test_image_centre_is_left_alone():
    assert ticked().getpixel((200, 150)) == (0, 0, 0)
