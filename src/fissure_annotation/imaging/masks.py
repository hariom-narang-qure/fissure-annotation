"""Put per-slice masks back into a 3D mask in the space of the scan they were cut from.

Slices are cut from the RAS-reoriented volume (see slices.py), so going back means:
- Undo the radiological flip and transpose of each slice, and put it back at its index along the axis of its view, which gives the mask in the RAS volume.
- A mask is one voxel thick along that axis.
- Masks of several views are combined with a voxel-wise OR, so a voxel is in the mask when any view marked it.
- Reorient the RAS volume to the orientation of the source file, so the mask lines up with the scan in any viewer.
"""

from collections.abc import Mapping
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Bool, UInt8
from nibabel import orientations

from fissure_annotation.imaging.slices import View, plane_selector


def ras_volume_from_slices(
    masks: Mapping[int, Bool[np.ndarray, "rows columns"]],
    shape_ras: tuple[int, int, int],
    view: View,
) -> UInt8[np.ndarray, "x y z"]:
    "slices without a mask stay empty"
    volume = np.zeros(shape_ras, dtype=np.uint8)
    for index, mask in masks.items():
        volume[plane_selector(view, index)] = mask[::-1, ::-1].T
    return volume


def reorient_to_source(
    volume_ras: UInt8[np.ndarray, "x y z"], source: nib.Nifti1Image
) -> nib.Nifti1Image:
    canonical = nib.as_closest_canonical(source)
    transform = orientations.ornt_transform(
        orientations.io_orientation(canonical.affine),
        orientations.io_orientation(source.affine),
    )
    image = nib.Nifti1Image(
        orientations.apply_orientation(volume_ras, transform),
        source.affine,
        header=source.header,
    )
    image.set_data_dtype(np.uint8)
    return image


def save_mask_like(
    source_path: Path,
    masks_by_view: Mapping[View, Mapping[int, Bool[np.ndarray, "rows columns"]]],
    out_path: Path,
) -> None:
    source = nib.load(source_path)
    shape_ras = nib.as_closest_canonical(source).shape
    volume = np.zeros(shape_ras, dtype=np.uint8)
    for view, masks in masks_by_view.items():
        volume |= ras_volume_from_slices(masks, shape_ras, view)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(reorient_to_source(volume, source), out_path)
