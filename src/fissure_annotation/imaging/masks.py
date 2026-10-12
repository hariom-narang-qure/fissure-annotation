"""Put per-slice masks back into a 3D mask in the space of the scan they were cut from.

Slices are cut from the RAS-reoriented volume (see slices.py), so going back means:
- Undo the radiological flip and transpose of each slice, which puts the mask in the RAS volume.
- Reorient the RAS volume to the orientation of the source file, so the mask lines up with the scan in any viewer.
"""
from collections.abc import Mapping
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Bool, UInt8
from nibabel import orientations


def ras_volume_from_slices(masks: Mapping[int, Bool[np.ndarray, "y x"]], shape_ras: tuple[int, int, int]) -> UInt8[np.ndarray, "x y z"]:
    "slices without a mask stay empty"
    volume = np.zeros(shape_ras, dtype=np.uint8)
    for z, mask in masks.items():
        volume[:, :, z] = mask[::-1, ::-1].T
    return volume


def reorient_to_source(volume_ras: UInt8[np.ndarray, "x y z"], source: nib.Nifti1Image) -> nib.Nifti1Image:
    canonical = nib.as_closest_canonical(source)
    transform = orientations.ornt_transform(orientations.io_orientation(canonical.affine), orientations.io_orientation(source.affine))
    image = nib.Nifti1Image(orientations.apply_orientation(volume_ras, transform), source.affine, header=source.header)
    image.set_data_dtype(np.uint8)
    return image


def save_mask_like(source_path: Path, masks: Mapping[int, Bool[np.ndarray, "y x"]], out_path: Path) -> None:
    source = nib.load(source_path)
    shape_ras = nib.as_closest_canonical(source).shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(reorient_to_source(ras_volume_from_slices(masks, shape_ras), source), out_path)
