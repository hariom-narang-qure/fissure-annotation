import nibabel as nib
import numpy as np

from fissure_annotation.imaging.slices import LUNG_WINDOW_HU, axial_view, to_lung_window, write_slices
from fissure_annotation.imaging.png import read_gray


def test_the_lung_window_maps_its_ends_to_black_and_white():
    low, high = LUNG_WINDOW_HU
    assert to_lung_window(np.array([low - 500, low, high, high + 500])).tolist() == [0, 0, 255, 255]


def test_patient_right_is_on_the_image_left_and_anterior_on_top():
    volume = np.zeros((20, 10, 3), dtype=np.float32)
    volume[18, 8, 1] = 1  # high x is patient right and high y is anterior in RAS
    row, column = np.argwhere(axial_view(volume, 1) == 1)[0]
    assert (row, column) == (1, 1)


def test_write_slices_writes_one_png_per_axial_slice(tmp_path):
    nifti = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.full((12, 10, 4), -850, dtype=np.int16), np.eye(4)), nifti)
    assert write_slices(nifti, tmp_path / "slices") == 4
    assert sorted(p.name for p in (tmp_path / "slices").iterdir()) == [f"slice_000{z}.png" for z in range(4)]
    assert read_gray(tmp_path / "slices" / "slice_0000.png").shape == (10, 12)
