from pathlib import Path

import numpy as np
from jaxtyping import UInt8
from PIL import Image


def read_gray(path: Path) -> UInt8[np.ndarray, "h w"]:
    return np.asarray(Image.open(path).convert("L"))


def save_png(image: UInt8[np.ndarray, "h w *channels"], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
