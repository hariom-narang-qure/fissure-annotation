import numpy as np
from PIL import Image

from fissure_annotation.drawing.lobes import LOBE_PAIRS, PAIR_COLOURS, draw_lobe_boundaries

RIGHT_UPPER_MIDDLE = LOBE_PAIRS.index((3, 4))


def two_lobes(size: int = 20) -> np.ndarray:
    lobes = np.zeros((size, size), dtype=np.uint8)
    lobes[:, : size // 2] = 3
    lobes[:, size // 2 :] = 4
    return lobes


def test_boundary_is_blended_not_painted_over():
    img = Image.new("RGB", (20, 20), (100, 100, 100))
    draw_lobe_boundaries(img, two_lobes())
    colour = PAIR_COLOURS[RIGHT_UPPER_MIDDLE]
    assert img.getpixel((10, 5)) == tuple((100 + c) // 2 for c in colour)
    assert img.getpixel((2, 5)) == (100, 100, 100)


def test_legend_names_the_colour_and_the_lobe_pair():
    img = Image.new("RGB", (20, 20))
    assert draw_lobe_boundaries(img, two_lobes()) == ["violet = right upper lobe / right middle lobe"]


def test_tiny_boundaries_are_skipped():
    lobes = np.zeros((20, 20), dtype=np.uint8)
    lobes[:2, :2] = 3
    lobes[:2, 2:4] = 4
    img = Image.new("RGB", (20, 20))
    assert draw_lobe_boundaries(img, lobes) == []


def test_boundary_scales_with_the_image_size():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    draw_lobe_boundaries(img, two_lobes())
    assert img.getpixel((50, 50)) != (0, 0, 0)
    assert img.getpixel((10, 50)) == (0, 0, 0)

