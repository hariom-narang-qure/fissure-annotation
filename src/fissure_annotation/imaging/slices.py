"""Cut a CT volume into the axial PNG series the model looks at.

Steps, and why:
- Reorient to RAS, then reverse x and y: the patient's right ends up on the image left and anterior on top, the usual radiological display.
- Smooth the volume in 3D (sigma 0.8, 0.8, 1.0 voxels): fissures are 1-2 px wide and thin slices are noisy, so light smoothing makes them readable without erasing them.
- Apply a narrow lung window of -1100 to -500 HU: parenchyma is about -850 HU, a fissure only slightly brighter, so a wide window flattens the contrast.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Float, UInt8
from scipy import ndimage

from fissure_annotation.imaging.png import save_png

LUNG_WINDOW_HU = (-1100.0, -500.0)
SMOOTHING_SIGMA_VOXELS = (0.8, 0.8, 1.0)


def load_volume_ras(path: Path) -> Float[np.ndarray, "x y z"]:
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


def to_lung_window(hu: Float[np.ndarray, "..."]) -> UInt8[np.ndarray, "..."]:
    low, high = LUNG_WINDOW_HU
    return (np.clip((hu - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def axial_view(volume: Float[np.ndarray, "x y z"], z: int) -> Float[np.ndarray, "y x"]:
    return volume[:, :, z].T[::-1, ::-1]


def slice_file_name(z: int) -> str:
    return f"slice_{z:04d}.png"


def write_slices(nifti: Path, out_dir: Path) -> int:
    volume = ndimage.gaussian_filter(load_volume_ras(nifti), SMOOTHING_SIGMA_VOXELS)
    for z in range(volume.shape[2]):
        save_png(to_lung_window(axial_view(volume, z)), out_dir / slice_file_name(z))
    return volume.shape[2]
