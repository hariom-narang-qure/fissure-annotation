"""One PNG per fissure showing a mask on the fissure as a wall, see imaging/wall.py for how the wall is made.

- Each mask voxel is given to the fissure whose lobe interface (the prior) is nearest, if within ASSIGN_MM.
  - The prior is a little off near fissures, so the tolerance is wide and the prior only decides which wall a voxel belongs to.
  - The wall itself is fitted to the mask voxels, not to the prior.
- The picture has three panels of the same wall: the CT, the CT with the mask in red, and the mask alone coloured by its offset from the fitted surface.
- Panel axes are the principal directions of the fissure, and each is labelled with the anatomical direction it mostly points to.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Float, UInt8
from PIL import Image, ImageDraw
from scipy import ndimage

from fissure_annotation.imaging.overlay import labelled
from fissure_annotation.imaging.png import save_png
from fissure_annotation.imaging.slices import load_volume_ras
from fissure_annotation.imaging.wall import MASK_REACH_MM, FissureWall, WallImages
from fissure_annotation.settings import ExperimentSettings

ASSIGN_MM = 10.0
FISSURE_NAMES = {1: "left_oblique", 2: "right_oblique", 3: "right_horizontal"}  # the label values of the TotalSegmentator lobe interfaces
UPSCALE = 3
RAS_DIRECTIONS = ("R", "A", "S")


class WallViewJob(ExperimentSettings):
    nifti: Path
    mask: Path
    fissure_prior: Path  # lobe interfaces, 1 left oblique, 2 right oblique, 3 right horizontal, same grid as the scan
    out_dir: Path


def spacing_ras_mm(nifti: Path) -> tuple[float, float, float]:
    return tuple(float(s) for s in nib.as_closest_canonical(nib.load(nifti)).header.get_zooms()[:3])


def mask_points_mm(path: Path, spacing: tuple[float, float, float]) -> Float[np.ndarray, "n 3"]:
    mask = load_volume_ras(path) > 0
    return np.argwhere(mask) * np.array(spacing)


def fissure_of_each_point(
    points_mm: Float[np.ndarray, "n 3"], prior_path: Path, spacing: tuple[float, float, float]
) -> np.ndarray:
    prior = np.asarray(nib.as_closest_canonical(nib.load(prior_path)).dataobj)
    distance, nearest = ndimage.distance_transform_edt(prior == 0, sampling=spacing, return_indices=True)
    voxels = np.round(points_mm / np.array(spacing)).astype(int)
    nearest_label = prior[tuple(nearest[:, voxels[:, 0], voxels[:, 1], voxels[:, 2]])]
    return np.where(distance[tuple(voxels.T)] <= ASSIGN_MM, nearest_label, 0)


def axis_name(axis: Float[np.ndarray, "3"]) -> str:
    dominant = int(np.argmax(np.abs(axis)))
    sign = "+" if axis[dominant] > 0 else "-"
    return f"{sign}{RAS_DIRECTIONS[dominant]}"


def offset_colours(offset_mm: Float[np.ndarray, "v u"]) -> UInt8[np.ndarray, "v u 3"]:
    "blue is on one side of the surface and red on the other, white is on it, black is no mask"
    out = np.zeros((*offset_mm.shape, 3), dtype=np.uint8)
    present = ~np.isnan(offset_mm)
    t = np.clip(np.nan_to_num(offset_mm) / MASK_REACH_MM, -1, 1)
    fade = ((1 - np.abs(t)) * 255).astype(np.uint8)
    out[..., 0] = np.where(t > 0, 255, fade)
    out[..., 2] = np.where(t < 0, 255, fade)
    out[..., 1] = fade
    out[~present] = 0
    return out


def red_on_ct(images: WallImages) -> UInt8[np.ndarray, "v u 3"]:
    rgb = np.stack([images.ct] * 3, axis=-1)
    rgb[~np.isnan(images.mask_offset_mm)] = (255, 0, 0)
    return rgb


def upscaled(tile: UInt8[np.ndarray, "v u 3"]) -> UInt8[np.ndarray, "V U 3"]:
    return np.kron(tile, np.ones((UPSCALE, UPSCALE, 1), dtype=np.uint8))


def wall_picture(images: WallImages, title: str) -> UInt8[np.ndarray, "h w 3"]:
    ct = np.stack([images.ct] * 3, axis=-1)
    tiles = [upscaled(t) for t in (ct, red_on_ct(images), offset_colours(images.mask_offset_mm))]
    names = ("CT", "CT + mask", f"mask offset, +-{MASK_REACH_MM:g} mm")
    return np.hstack([labelled(t, f"{title}: {n}") for t, n in zip(tiles, names)])


def make_wall_views(job: WallViewJob) -> list[Path]:
    spacing = spacing_ras_mm(job.nifti)
    volume = load_volume_ras(job.nifti)
    points = mask_points_mm(job.mask, spacing)
    fissure = fissure_of_each_point(points, job.fissure_prior, spacing)
    written = []
    for label, name in FISSURE_NAMES.items():
        own = points[fissure == label]
        if len(own) == 0:
            continue
        wall = FissureWall(own)
        images = wall.flatten(volume, spacing, own)
        title = f"{name} (u {axis_name(wall.axes[0])}, v {axis_name(wall.axes[1])})"
        path = job.out_dir / f"wall_{name}.png"
        save_png(wall_picture(images, title), path)
        written.append(path)
    return written
