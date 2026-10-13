import nibabel as nib
import numpy as np
import pytest

from fissure_annotation.imaging.masks import save_mask_like
from fissure_annotation.imaging.slices import VIEW_AXIS, slice_view, load_volume_ras

MARKER = (2, 3, 4)
AFFINES = {
    "ras": np.eye(4),
    "flipped x and y": np.diag([-1.0, -1.0, 1.0, 1.0]),
    "axes swapped": np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]),
}


@pytest.mark.parametrize("view", VIEW_AXIS)
@pytest.mark.parametrize("affine", AFFINES.values(), ids=AFFINES.keys())
def test_a_marked_voxel_comes_back_at_the_same_place_in_the_source_array(tmp_path, affine, view):
    "marks the voxel on the slice as the model would see it, then checks the mask has it where the voxel is in the source file"
    marked = np.zeros((6, 7, 8), dtype=np.int16)
    marked[MARKER] = 1000
    source = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(marked, affine), source)
    volume_ras = load_volume_ras(source)
    masks = {i: slice_view(volume_ras, view, i) > 500 for i in range(volume_ras.shape[VIEW_AXIS[view]])}
    save_mask_like(source, {view: masks}, tmp_path / "mask.nii.gz")
    mask = nib.load(tmp_path / "mask.nii.gz")
    assert np.argwhere(np.asarray(mask.dataobj)).tolist() == [list(MARKER)]
    assert np.allclose(mask.affine, affine) and mask.get_data_dtype() == np.uint8


def test_slices_without_a_mask_stay_empty(tmp_path):
    source = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 3), dtype=np.int16), np.eye(4)), source)
    save_mask_like(source, {"axial": {1: np.ones((5, 4), dtype=bool)}}, tmp_path / "mask.nii.gz")
    mask = np.asarray(nib.load(tmp_path / "mask.nii.gz").dataobj)
    assert mask[:, :, 1].all() and not mask[:, :, 0].any() and not mask[:, :, 2].any()


def test_masks_of_several_views_are_combined_with_or(tmp_path):
    source = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 3), dtype=np.int16), np.eye(4)), source)
    axial_slice = np.zeros((5, 4), dtype=bool)
    axial_slice[0, 0] = True
    coronal_slice = np.zeros((3, 4), dtype=bool)
    coronal_slice[0, 0] = True
    save_mask_like(source, {"axial": {0: axial_slice}, "coronal": {2: coronal_slice}}, tmp_path / "mask.nii.gz")
    mask = np.asarray(nib.load(tmp_path / "mask.nii.gz").dataobj)
    assert mask.sum() == 2  # two different voxels, one from each view
