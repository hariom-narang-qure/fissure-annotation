from PIL import Image

from fissure_annotation.drawing.panels import PANEL_GAP, contact_sheet, labelled_thumbnail, side_by_side, titled_row


def solid(w: int, h: int, colour) -> Image.Image:
    return Image.new("RGB", (w, h), colour)


def test_titled_row_makes_room_for_titles_above_the_tallest_tile():
    out = titled_row([("a", solid(10, 20, "red")), ("b", solid(30, 10, "blue"))], title_height=18)
    assert out.size == (10 + 30 + 2 * PANEL_GAP, 20 + 18)
    assert out.getpixel((5, 18 + 5)) == (255, 0, 0)


def test_side_by_side_puts_the_second_image_after_a_gap():
    out = side_by_side(solid(10, 10, "red"), solid(10, 10, "blue"), "t")
    assert out.size == (10 + 10 + PANEL_GAP, 10 + 20)
    assert out.getpixel((10 + PANEL_GAP + 5, 25)) == (0, 0, 255)


def test_thumbnail_scales_the_long_side():
    tile = labelled_thumbnail(solid(200, 100, "black"), long_side=50, label="x")
    assert tile.size == (50, 25)


def test_contact_sheet_reads_left_to_right_then_down():
    tiles = [solid(10, 10, c) for c in ("red", "lime", "blue", "yellow", "white")]
    sheet = contact_sheet(tiles, columns=2)
    assert sheet.size == (20, 30)
    assert sheet.getpixel((15, 5)) == (0, 255, 0)
    assert sheet.getpixel((5, 25)) == (255, 255, 255)
    assert sheet.getpixel((15, 25)) == (0, 0, 0)
