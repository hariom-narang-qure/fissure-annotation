"""The three coordinate frames the model, the tools and the slice use, and the maps between them.

- Slice pixels: (col, row) in the full isotropic slice.
- Main frame: 0-1000 in x and y over the crop box, the frame of the first image the model sees.
- View frame: 0-1000 over a view, which is a box inside the main frame (a zoom).
  - Views nest, so a zoom of a zoom is still a box in the main frame.
  - Lines are always stored in the main frame, so a line survives any change of view.
"""
from dataclasses import dataclass

PixelBox = tuple[int, int, int, int]
MainBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CropFrame:
    "slice pixels <-> main frame, for the crop box (col0, row0, col1, row1) shown as the first image"

    box: PixelBox

    def to_main(self, col: float, row: float) -> tuple[int, int]:
        c0, r0, c1, r1 = self.box
        return round((col - c0) / (c1 - c0) * 1000), round((row - r0) / (r1 - r0) * 1000)

    def to_pixels(self, x: float, y: float) -> tuple[float, float]:
        c0, r0, c1, r1 = self.box
        return c0 + x / 1000 * (c1 - c0), r0 + y / 1000 * (r1 - r0)


def view_to_main(x: float, y: float, view_box: MainBox) -> tuple[float, float]:
    x0, y0, x1, y1 = view_box
    return x0 + x / 1000 * (x1 - x0), y0 + y / 1000 * (y1 - y0)


def main_to_view(x: float, y: float, view_box: MainBox) -> tuple[float, float]:
    x0, y0, x1, y1 = view_box
    return (x - x0) / (x1 - x0) * 1000, (y - y0) / (y1 - y0) * 1000
