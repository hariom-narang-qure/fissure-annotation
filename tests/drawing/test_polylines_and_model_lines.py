from PIL import Image

from fissure_annotation.drawing.model_lines import CONFIDENCE_COLOURS, draw_model_lines
from fissure_annotation.lines import ProposedFissure, Waypoint
from fissure_annotation.drawing.polylines import POLYLINE_COLOURS, draw_polylines


def fissure(confidence: str, y: int) -> ProposedFissure:
    return ProposedFissure(name="oblique", side="right", confidence=confidence, waypoints=[Waypoint(x=100, y=y), Waypoint(x=900, y=y)])


def test_model_lines_are_coloured_by_confidence():
    out = draw_model_lines(Image.new("L", (200, 200)), [fissure("clear", 250), fissure("faint", 750)])
    assert out.getpixel((100, 50)) == CONFIDENCE_COLOURS["clear"]
    assert out.getpixel((100, 150)) == CONFIDENCE_COLOURS["faint"]


def test_model_lines_leave_the_input_untouched():
    img = Image.new("RGB", (200, 200))
    draw_model_lines(img, [fissure("clear", 500)])
    assert img.getpixel((100, 100)) == (0, 0, 0)


def test_uncertain_still_draws():
    out = draw_model_lines(Image.new("L", (200, 200)), [fissure("uncertain", 500)])
    assert out.getpixel((100, 100)) != (0, 0, 0)


def test_polylines_are_given_in_the_0_to_1000_frame_and_cycle_colours():
    img = Image.new("RGB", (100, 100))
    first_row_of_each_colour = len(POLYLINE_COLOURS)
    lines = [[(0, 100 * k), (1000, 100 * k)] for k in range(first_row_of_each_colour + 1)]
    draw_polylines(img, lines)
    assert img.getpixel((50, 10)) == (0, 255, 0)  # second line, y = 100 of 1000 on a 100 px image
    assert img.getpixel((50, 70)) == img.getpixel((50, 0)) == (255, 0, 0)  # the colours wrapped round
