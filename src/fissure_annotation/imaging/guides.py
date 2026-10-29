"""Dashed outlines of the fissure bands, drawn on a slice for the model to see.

Why:
- The model could not relate a separate band mask image to the slice.
- An outline on the slice itself puts the two edges of a band right where it looks.

How:
- The outline of a band is its pixels that touch a pixel outside it (one erosion).
  - Along a fissure that gives two lines about a band width apart, closed where the band ends.
- The outline is dashed and coloured, so it cannot pass for a fissure, which is a thin bright grey line.
- Only the picture is drawn on. The tracer keeps following the ridge of the clean slice.
"""

import numpy as np
from jaxtyping import Bool, UInt8
from scipy import ndimage

DASH_PX = 3  # pixels on, then pixels off, along the diagonal pattern of the dash

Colour = tuple[int, int, int]


def outline_of(region: Bool[np.ndarray, "h w"]) -> Bool[np.ndarray, "h w"]:
    return region & ~ndimage.binary_erosion(region)


def dashed(mask: Bool[np.ndarray, "h w"]) -> Bool[np.ndarray, "h w"]:
    "keeps every other run of DASH_PX diagonals, which breaks any line into dashes whatever its direction"
    rows, columns = np.indices(mask.shape)
    return mask & (((rows + columns) // DASH_PX) % 2 == 0)


def draw_band_outlines(
    picture: UInt8[np.ndarray, "h w *channels"],
    bands: UInt8[np.ndarray, "h w"],
    colour_of_value: dict[int, Colour],
) -> UInt8[np.ndarray, "h w 3"]:
    "bands holds one pixel value per fissure, 0 outside; a grey picture is made RGB"
    rgb = np.stack([picture] * 3, axis=-1) if picture.ndim == 2 else picture.copy()
    for value, colour in colour_of_value.items():
        rgb[dashed(outline_of(bands == value))] = colour
    return rgb
