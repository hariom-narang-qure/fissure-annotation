from PIL import Image, ImageDraw

TICK_COLOUR = (255, 255, 0)
TICK_EVERY = 100
TICK_LENGTH = 8


def draw_ticks(img: Image.Image) -> None:
    """Tick marks every 100 units on all edges, labels on the left and top edges.

    - Drawn inside the image, never in a margin, so the image frame is exactly the 0-1000 frame.
    - A margin layout made it ambiguous which pixel the model's coordinates referred to.
    """
    d = ImageDraw.Draw(img)
    w, h = img.size
    for v in range(0, 1001, TICK_EVERY):
        x, y = v / 1000 * (w - 1), v / 1000 * (h - 1)
        for xy in tick_segments(x, y, w, h):
            d.line(xy, fill=TICK_COLOUR, width=1)
        if 0 < v < 1000:
            d.text((x + 2, 9), str(v), fill=TICK_COLOUR)
            d.text((10, y - 5), str(v), fill=TICK_COLOUR)


def tick_segments(x: float, y: float, w: int, h: int) -> list[list[tuple[float, float]]]:
    n = TICK_LENGTH
    return [[(x, 0), (x, n)], [(x, h - 1), (x, h - 1 - n)], [(0, y), (n, y)], [(w - 1, y), (w - 1 - n, y)]]
