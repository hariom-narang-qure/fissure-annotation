from PIL import Image, ImageDraw

Polyline = list[tuple[int, int]]

POLYLINE_COLOURS = ["red", "lime", "cyan", "magenta", "orange", "yellow", "deepskyblue"]


def draw_polylines(img: Image.Image, polylines: list[Polyline]) -> None:
    "polylines given in the 0-1000 frame of `img`, one colour each, cycling"
    d = ImageDraw.Draw(img)
    for k, points in enumerate(polylines):
        px = [(x / 1000 * img.width, y / 1000 * img.height) for x, y in points]
        d.line(px, fill=POLYLINE_COLOURS[k % len(POLYLINE_COLOURS)], width=2)
