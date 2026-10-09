from PIL import Image, ImageDraw

COLOURS = {"clear": "red", "faint": "yellow", "uncertain": "orange"}


def to_pixels(wp: dict, size: tuple[int, int]) -> tuple[float, float]:
    "model coords are 0-1000 normalised, origin top-left"
    w, h = size
    return wp["x"] / 1000 * w, wp["y"] / 1000 * h


def draw_fissures(img: Image.Image, proposal: dict) -> Image.Image:
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    for f in proposal["fissures"]:
        pts = [to_pixels(wp, img.size) for wp in f["waypoints"]]
        colour = COLOURS.get(f["confidence"], "orange")
        d.line(pts, fill=colour, width=1)
        for x, y in pts:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], outline=colour)
    return out
