import nibabel as nib
import numpy as np

from fissure_annotation.imaging.resample import (
    resample_z_to_pixel_spacing,
    resampled_slice_count,
)


def test_the_first_and_last_slices_keep_their_place_in_the_world(tmp_path):
    affine = np.diag([0.5, 0.5, 2.0, 1.0])
    affine[:3, 3] = (10.0, 20.0, 30.0)
    volume = np.zeros((4, 4, 3), dtype=np.int16)
    volume[..., 2] = 100
    nib.save(nib.Nifti1Image(volume, affine), tmp_path / "in.nii.gz")
    resample_z_to_pixel_spacing(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz")
    out = nib.load(tmp_path / "out.nii.gz")
    last = out.shape[2] - 1
    assert out.shape[2] == 9  # 2 gaps of 2 mm become 8 gaps of 0.5 mm
    assert np.allclose(out.affine @ [0, 0, last, 1], affine @ [0, 0, 2, 1])
    assert np.asarray(out.dataobj)[0, 0, last] == 100


def test_values_between_slices_are_interpolated(tmp_path):
    volume = np.zeros((2, 2, 2), dtype=np.int16)
    volume[..., 1] = 100
    nib.save(nib.Nifti1Image(volume, np.diag([1.0, 1.0, 2.0, 1.0])), tmp_path / "in.nii.gz")
    resample_z_to_pixel_spacing(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz")
    assert list(np.asarray(nib.load(tmp_path / "out.nii.gz").dataobj)[0, 0]) == [0, 50, 100]


def test_a_scan_is_never_given_fewer_slices():
    assert resampled_slice_count(10, 0.5) == 10
