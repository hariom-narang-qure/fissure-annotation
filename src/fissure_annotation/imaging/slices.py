"""Cut a CT volume into the PNG series (axial, coronal or sagittal) the model looks at.

Steps, and why:
- Reorient to RAS, so the three views are plain index planes of the array.
- Transpose and reverse both image axes of every plane (see slice_view), which gives the usual radiological display:
  - axial: anterior on top, patient right on the image left.
  - coronal: superior on top, patient right on the image left.
  - sagittal: superior on top, anterior on the image left.
- Planes are cut on the voxel grid, so coronal and sagittal images are stretched or squashed along z when the slice spacing is not the in-plane spacing.
- Smooth the volume in 3D (sigma 0.8, 0.8, 1.0 voxels): fissures are 1-2 px wide and thin slices are noisy, so light smoothing makes them readable without erasing them.
- Apply a narrow lung window of -1100 to -500 HU: parenchyma is about -850 HU, a fissure only slightly brighter, so a wide window flattens the contrast.
"""

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from jaxtyping import Float, UInt8
from scipy import ndimage

from fissure_annotation.imaging.png import save_png

LUNG_WINDOW_HU = (-1100.0, -500.0)
SMOOTHING_SIGMA_VOXELS = (0.8, 0.8, 1.0)

View = Literal["axial", "coronal", "sagittal"]
VIEW_AXIS = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}  # the RAS array axis a view steps through


def load_volume_ras(path: Path) -> Float[np.ndarray, "x y z"]:
    return np.asarray(
        nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32
    )


def to_lung_window(hu: Float[np.ndarray, "..."]) -> UInt8[np.ndarray, "..."]:
    low, high = LUNG_WINDOW_HU
    return (np.clip((hu - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def plane_selector(view: View, index: int) -> tuple:
    selector = [slice(None)] * 3
    selector[VIEW_AXIS[view]] = index
    return tuple(selector)


def slice_view(
    volume: Float[np.ndarray, "x y z"], view: View, index: int
) -> Float[np.ndarray, "rows columns"]:
    return volume[plane_selector(view, index)].T[::-1, ::-1]


def slice_file_name(z: int) -> str:
    return f"slice_{z:04d}.png"


def write_slices(nifti: Path, out_dir: Path, view: View = "axial") -> int:
    volume = ndimage.gaussian_filter(load_volume_ras(nifti), SMOOTHING_SIGMA_VOXELS)
    count = volume.shape[VIEW_AXIS[view]]
    for index in range(count):
        save_png(
            to_lung_window(slice_view(volume, view, index)),
            out_dir / slice_file_name(index),
        )
    return count


def view_pixel_spacing(spacing_xyz: tuple[float, float, float], view: View) -> tuple[float, float]:
    "rows then columns of a slice image, which slice_view makes the second then the first in-plane RAS axis"
    first, second = (axis for axis in range(3) if axis != VIEW_AXIS[view])
    return spacing_xyz[second], spacing_xyz[first]


def write_label_slices(
    labels: UInt8[np.ndarray, "x y z"], out_dir: Path, view: View, grey_of_label: dict[int, int]
) -> int:
    "labels in RAS, cut like write_slices but without smoothing or window, each label drawn at its grey level"
    grey = np.zeros(256, dtype=np.uint8)
    for label, level in grey_of_label.items():
        grey[label] = level
    count = labels.shape[VIEW_AXIS[view]]
    for index in range(count):
        save_png(grey[slice_view(labels, view, index)], out_dir / slice_file_name(index))
    return count
