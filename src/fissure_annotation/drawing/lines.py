from PIL import Image, ImageDraw

Point = tuple[float, float]

LINE_COLOURS = [(255, 40, 40), (255, 220, 0), (0, 255, 120), (0, 200, 255), (255, 0, 255), (255, 140, 0)]
OWN_WAYPOINTS_COLOUR = (255, 255, 255)
POINT_RADIUS = 3


def line_colour(k: int) -> tuple[int, int, int]:
    return LINE_COLOURS[k % len(LINE_COLOURS)]


def draw_traced_line(img: Image.Image, own_waypoints: list[Point], traced: list[Point], colour, label: str) -> None:
    """The traced path thick and coloured, the model's own waypoints thin and white.

    - Both are shown so the model can see how far the tracer moved its points.
    - Points are in image pixels.
    """
    d = ImageDraw.Draw(img)
    d.line(own_waypoints, fill=OWN_WAYPOINTS_COLOUR, width=1)
    d.line(traced, fill=colour, width=2)
    draw_label(d, traced[0], label, colour)


def draw_marked_line(img: Image.Image, points: list[Point], colour, label: str) -> None:
    "no tracer: the line is the model's points joined by straight segments, with a circle at each point"
    d = ImageDraw.Draw(img)
    d.line(points, fill=colour, width=2)
    for x, y in points:
        d.ellipse([x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS], outline=colour)
    draw_label(d, points[0], label, colour)


def draw_label(d: ImageDraw.ImageDraw, start: Point, label: str, colour) -> None:
    d.text((start[0] + 5, start[1] - 14), label, fill=colour)
