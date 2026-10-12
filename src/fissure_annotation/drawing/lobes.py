from itertools import combinations

import numpy as np
from PIL import Image

from fissure_annotation.slicing import LOBE_NAMES, boundary_pair

LOBE_LABELS = range(1, 6)
LOBE_PAIRS = list(combinations(LOBE_LABELS, 2))
PAIR_COLOURS = [(255, 60, 60), (60, 255, 60), (60, 200, 255), (255, 0, 255), (255, 160, 0), (255, 255, 0), (0, 255, 200), (180, 120, 255), (255, 120, 180), (160, 255, 60)]
COLOUR_NAMES = ["red", "green", "light blue", "magenta", "orange", "yellow", "turquoise", "violet", "pink", "lime"]
MIN_BOUNDARY_PIXELS = 6


def draw_lobe_boundaries(img: Image.Image, lobes: np.ndarray) -> list[str]:
    """Blend a colour onto every pixel where two lobes touch, and return the colour legend.

    - `lobes` is the label array of the region shown, at the resolution of the source slice.
    - Each lobe pair has a fixed colour, so the legend wording is stable across calls and slices.
    - Blended half and half, so a fissure under the boundary stays visible.
    - Pairs with fewer than 6 boundary pixels are skipped, they are rounding noise from the segmentation.
    """
    legend = []
    px = img.load()
    for k, (a, b) in enumerate(LOBE_PAIRS):
        boundary = boundary_pair(lobes, a, b)
        if boundary.sum() < MIN_BOUNDARY_PIXELS:
            continue
        for y, x in zip(*np.nonzero(upscale_mask(boundary, img.size))):
            px[int(x), int(y)] = blend(px[int(x), int(y)], PAIR_COLOURS[k])
        legend.append(f"{COLOUR_NAMES[k]} = {LOBE_NAMES[a]} / {LOBE_NAMES[b]}")
    return legend


def upscale_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


def blend(pixel: tuple[int, ...], colour: tuple[int, int, int]) -> tuple[int, ...]:
    return tuple((o + c) // 2 for o, c in zip(pixel, colour))
