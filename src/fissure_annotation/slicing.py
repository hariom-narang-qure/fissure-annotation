"""Cut one isotropic, radiological-orientation lung-window slice (plus matching lobe labels) from a CT volume.

Display convention (what the prompt tells the model): patient RIGHT is on the image LEFT; top is anterior
(axial) or superior (coronal). nibabel's closest-canonical gives RAS, where array x grows toward the patient's
right, so x is reversed here. v1 (`slice.py`) did not reverse it and sent mirrored slices.

In-plane pixels are resampled to be square (mm) so a 0.3 mm-slice scan is not shown stretched.
"""
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

LUNG_CENTRE, LUNG_WIDTH = -600, 1500
LOBE_FILES = ["left_upper_lobe", "left_lower_lobe", "right_upper_lobe", "right_middle_lobe", "right_lower_lobe"]
LOBE_NAMES = {i + 1: n.replace("_", " ") for i, n in enumerate(LOBE_FILES)}


@dataclass
class Case:
    vol: np.ndarray                   # RAS (x, y, z) HU
    lobes: np.ndarray | None          # same grid, 0 = none, 1..5 = LOBE_FILES order
    spacing: tuple[float, float, float]


@dataclass
class View:
    image: np.ndarray                 # uint8, lung window
    lobes: np.ndarray | None          # same shape as image
    mm_per_px: float
    plane: str
    index: int
    hu: np.ndarray | None = None      # same shape as image, float HU, for re-windowing


def load_case(scan_path: str, lobe_dir: str | None = None) -> Case:
    nii = nib.as_closest_canonical(nib.load(scan_path))
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    lobes = load_lobes(lobe_dir, vol.shape) if lobe_dir else None
    return Case(vol, lobes, tuple(float(z) for z in nii.header.get_zooms()[:3]))


def load_lobes(lobe_dir: str, shape) -> np.ndarray | None:
    out = np.zeros(shape, dtype=np.uint8)
    for label, name in enumerate(LOBE_FILES, start=1):
        f = Path(lobe_dir) / f"{name}.nii.gz"
        if f.exists():
            m = np.asarray(nib.as_closest_canonical(nib.load(str(f))).dataobj) > 0
            out[m] = label
    return out


def lung_window(hu: np.ndarray, centre: float = LUNG_CENTRE, width: float = LUNG_WIDTH) -> np.ndarray:
    lo = centre - width / 2
    return (np.clip((hu - lo) / width, 0, 1) * 255).astype(np.uint8)


def cut(case: Case, plane: str, index: int) -> View:
    sx, sy, sz = case.spacing
    axis, other_mm = (1, sz) if plane == "coronal" else (2, sy)
    take = (lambda a: a[:, index, :]) if plane == "coronal" else (lambda a: a[:, :, index])
    pix = max(sx, other_mm)                               # square pixels at the coarser spacing: never invent resolution
    zoom = (sx / pix, other_mm / pix)
    img = orient(ndimage.zoom(take(case.vol), zoom, order=1))
    lob = None if case.lobes is None else orient(ndimage.zoom(take(case.lobes), zoom, order=0))
    return View(lung_window(img), lob, pix, plane, index, hu=img)


def orient(a: np.ndarray) -> np.ndarray:
    "a is (x, z|y) in RAS: reverse both so patient right is image left and superior/anterior is up"
    return a[::-1, ::-1].T


def lung_extent(case: Case, axis: int) -> tuple[int, int]:
    "first/last index along `axis` (0=x,1=y,2=z) where any lobe voxel exists"
    present = np.nonzero((case.lobes > 0).any(axis=tuple(i for i in range(3) if i != axis)))[0]
    return int(present[0]), int(present[-1])


def boundary_mask(lobes: np.ndarray) -> np.ndarray:
    "pixels where two different non-zero lobes touch: the proxy for where a fissure is"
    out = np.zeros(lobes.shape, bool)
    for sl_a, sl_b in [((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                       ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))]:
        a, b = lobes[sl_a], lobes[sl_b]
        diff = (a != b) & (a > 0) & (b > 0)
        out[sl_a] |= diff
        out[sl_b] |= diff
    return out


def boundary_pair(lobes: np.ndarray, a: int, b: int) -> np.ndarray:
    "the boundary between lobes `a` and `b` only, ignoring every other lobe"
    return boundary_mask(np.where(np.isin(lobes, [a, b]), lobes, 0))


def save_png(img: np.ndarray, path) -> None:
    Image.fromarray(img).save(path)
