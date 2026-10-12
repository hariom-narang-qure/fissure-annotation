from PIL import Image, ImageDraw

Point = tuple[float, float]

RING_COLOUR = (255, 0, 255)
RING_RADIUS = 5


def draw_point_rings(img: Image.Image, points: list[Point], join_colour: tuple[int, int, int] | None = None) -> None:
    """Numbered rings at the given image pixels, in order, starting at 1.

    - Rings are hollow so the thin line under a point stays visible.
    - The line is meant to pass through every ring's centre.
    - `join_colour` also draws a thin line through the points, off by default since it hides a thin fissure.
    """
    d = ImageDraw.Draw(img)
    if join_colour is not None:
        d.line(points, fill=join_colour, width=1)
    for k, (x, y) in enumerate(points, start=1):
        d.ellipse([x - RING_RADIUS, y - RING_RADIUS, x + RING_RADIUS, y + RING_RADIUS], outline=RING_COLOUR, width=2)
        d.text((x + 8, y - 15), str(k), fill=RING_COLOUR)
