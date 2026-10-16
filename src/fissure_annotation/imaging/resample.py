"""Resample a CT scan along z so its voxels are about as tall as they are wide.

Why:
- Coronal and sagittal slices are cut on the voxel grid (see slices.py).
  - On a scan with 3 mm slices and 0.8 mm pixels the model saw lungs squashed about 3.7 times vertically.
- The slice step and the sheet blur count voxels, so on such a scan they also meant 3.7 times more along z than across.

How:
1. The scan is reoriented to RAS, so z is the third array axis.
2. Only z is resampled, to the in-plane pixel spacing, by linear interpolation.
   - The first and last slices keep their positions, so the new spacing is the old extent over the new slice count.
3. The z column of the affine is scaled by the same factor, so the scan stays in the same place in the world.
4. The HU values are rounded back to int16, which keeps the file the size of a 1 mm scan.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def slice_to_pixel_ratio(image: nib.Nifti1Image) -> float:
    "image must be in RAS, so the third spacing is the slice spacing"
    x_mm, _, z_mm = (float(s) for s in image.header.get_zooms()[:3])
    return z_mm / x_mm


def resampled_slice_count(slice_count: int, ratio: float) -> int:
    return max(round((slice_count - 1) * ratio) + 1, slice_count)


def affine_with_z_scaled(affine: np.ndarray, factor: float) -> np.ndarray:
    scaled = affine.copy()
    scaled[:3, 2] *= factor
    return scaled


def resample_z_to_pixel_spacing(source: Path, target: Path) -> None:
    image = nib.as_closest_canonical(nib.load(source))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    count = resampled_slice_count(volume.shape[2], slice_to_pixel_ratio(image))
    zoom = (1.0, 1.0, count / volume.shape[2])  # sets the output size, grid_mode=False keeps the end slices in place
    resampled = ndimage.zoom(volume, zoom, order=1, grid_mode=False)
    affine = affine_with_z_scaled(image.affine, (volume.shape[2] - 1) / (count - 1))
    out = nib.Nifti1Image(np.round(resampled).astype(np.int16), affine)
    out.set_data_dtype(np.int16)
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, target)
