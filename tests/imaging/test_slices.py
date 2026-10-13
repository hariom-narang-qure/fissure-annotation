import nibabel as nib
import numpy as np

from fissure_annotation.imaging.slices import LUNG_WINDOW_HU, slice_view, to_lung_window, write_slices
from fissure_annotation.imaging.png import read_gray


def test_the_lung_window_maps_its_ends_to_black_and_white():
    low, high = LUNG_WINDOW_HU
    assert to_lung_window(np.array([low - 500, low, high, high + 500])).tolist() == [0, 0, 255, 255]


def test_patient_right_is_on_the_image_left_and_anterior_on_top():
    volume = np.zeros((20, 10, 3), dtype=np.float32)
    volume[18, 8, 1] = 1  # high x is patient right and high y is anterior in RAS
    row, column = np.argwhere(slice_view(volume, "axial", 1) == 1)[0]
    assert (row, column) == (1, 1)


def test_coronal_has_superior_on_top_and_patient_right_on_the_image_left():
    volume = np.zeros((20, 10, 6), dtype=np.float32)
    volume[18, 4, 4] = 1  # high x is patient right and high z is superior in RAS
    row, column = np.argwhere(slice_view(volume, "coronal", 4) == 1)[0]
    assert (row, column) == (1, 1)


def test_sagittal_has_superior_on_top_and_anterior_on_the_image_left():
    volume = np.zeros((20, 10, 6), dtype=np.float32)
    volume[3, 8, 4] = 1
    row, column = np.argwhere(slice_view(volume, "sagittal", 3) == 1)[0]
    assert (row, column) == (1, 1)


def test_each_view_writes_one_png_per_index_of_its_axis(tmp_path):
    nifti = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.full((12, 10, 4), -850, dtype=np.int16), np.eye(4)), nifti)
    assert write_slices(nifti, tmp_path / "coronal", "coronal") == 10
    assert write_slices(nifti, tmp_path / "sagittal", "sagittal") == 12
    assert read_gray(tmp_path / "coronal" / "slice_0000.png").shape == (4, 12)
    assert read_gray(tmp_path / "sagittal" / "slice_0000.png").shape == (4, 10)


def test_write_slices_writes_one_png_per_axial_slice(tmp_path):
    nifti = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.full((12, 10, 4), -850, dtype=np.int16), np.eye(4)), nifti)
    assert write_slices(nifti, tmp_path / "slices") == 4
    assert sorted(p.name for p in (tmp_path / "slices").iterdir()) == [f"slice_000{z}.png" for z in range(4)]
    assert read_gray(tmp_path / "slices" / "slice_0000.png").shape == (10, 12)
