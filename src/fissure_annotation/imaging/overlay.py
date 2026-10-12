import numpy as np
from jaxtyping import Bool, UInt8
from PIL import Image, ImageDraw
from scipy import ndimage

RED = (255, 0, 0)
LABEL_COLOUR = (255, 255, 0)


def red_overlay(image: UInt8[np.ndarray, "h w"], mask: Bool[np.ndarray, "h w"]) -> UInt8[np.ndarray, "h w 3"]:
    "the mask is thickened by one pixel so a 1 px line stays visible"
    rgb = np.stack([image] * 3, axis=-1)
    rgb[ndimage.binary_dilation(mask)] = RED
    return rgb


def contact_sheet(tiles: list[UInt8[np.ndarray, "h w 3"]], columns: int) -> UInt8[np.ndarray, "H W 3"]:
    "tiles must share one shape, the last row is padded with black"
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (-len(tiles) % columns)
    rows = [np.hstack(padded[i:i + columns]) for i in range(0, len(padded), columns)]
    return np.vstack(rows)


def labelled(tile: UInt8[np.ndarray, "h w 3"], label: str) -> UInt8[np.ndarray, "h w 3"]:
    picture = Image.fromarray(tile)
    ImageDraw.Draw(picture).text((6, 6), label, fill=LABEL_COLOUR)
    return np.asarray(picture)
