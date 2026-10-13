"""A magnified crop with coordinate ticks, so a model can read pixel positions off the image.

Ticks sit in a margin outside the crop, so they never cover the anatomy.
Labels are in the pixel coordinates of the full slice, not of the crop.
"""

import numpy as np
from jaxtyping import UInt8
from PIL import Image, ImageDraw

TICK_EVERY_PX = 20
MARGIN_PX = 34
TICK_LENGTH_PX = 6
TICK_COLOUR = (255, 255, 0)

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in slice pixels


def crop_with_ticks(
    image: UInt8[np.ndarray, "h w *channels"], box: Box, scale: int
) -> Image.Image:
    x0, y0, x1, y1 = box
    crop = (
        Image.fromarray(image)
        .convert("RGB")
        .crop(box)
        .resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.LANCZOS)
    )
    canvas = Image.new("RGB", (crop.width + MARGIN_PX, crop.height + MARGIN_PX))
    canvas.paste(crop, (MARGIN_PX, MARGIN_PX))
    draw = ImageDraw.Draw(canvas)
    for x in tick_positions(x0, x1):
        at = MARGIN_PX + (x - x0) * scale
        draw.line([(at, MARGIN_PX - TICK_LENGTH_PX), (at, MARGIN_PX)], fill=TICK_COLOUR)
        draw.text((at - 10, 8), str(x), fill=TICK_COLOUR)
    for y in tick_positions(y0, y1):
        at = MARGIN_PX + (y - y0) * scale
        draw.line([(MARGIN_PX - TICK_LENGTH_PX, at), (MARGIN_PX, at)], fill=TICK_COLOUR)
        draw.text((1, at - 5), str(y), fill=TICK_COLOUR)
    return canvas


def tick_positions(start: int, stop: int) -> range:
    return range((start // TICK_EVERY_PX + 1) * TICK_EVERY_PX, stop, TICK_EVERY_PX)
