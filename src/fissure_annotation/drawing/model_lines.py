from PIL import Image, ImageDraw

CONFIDENCE_COLOURS = {"clear": (255, 40, 40), "faint": (255, 220, 0), "uncertain": (255, 140, 0)}
UNKNOWN_CONFIDENCE_COLOUR = (255, 140, 0)


def draw_model_lines(img: Image.Image, fissures: list[dict]) -> Image.Image:
    """The model's final fissures on a copy of the image, coloured by its confidence.

    - Waypoints are in the 0-1000 frame of `img`.
    - Each line is numbered, with its side and confidence, so a reviewer can refer to it.
    """
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    for k, f in enumerate(fissures, start=1):
        px = [(p["x"] / 1000 * img.width, p["y"] / 1000 * img.height) for p in f["waypoints"]]
        colour = CONFIDENCE_COLOURS.get(f["confidence"], UNKNOWN_CONFIDENCE_COLOUR)
        d.line(px, fill=colour, width=2)
        d.text((px[0][0] + 4, px[0][1] - 12), f"{k} {f['side'][0].upper()} {f['confidence']}", fill=colour)
    return out
