import nibabel as nib
import numpy as np
import pytest

from fissure_annotation.imaging.lobes import LEFT_LOWER, LEFT_UPPER
from fissure_annotation.postprocess import (
    PostProcessJob,
    SheetSettings,
    post_process_fissure_masks,
    sheet_parameters,
)
from fissure_annotation.segmentation import LUNG_LOBES

SIZE = 40
BOUNDARY_X = 20  # the left upper lobe is below it and the left lower lobe from it up
FAR_X = 2  # 8 mm outside the lungs, which start at x=10, and the margin is 5 mm
INSIDE_FAR_X = 12  # inside the lungs, 8 mm from the lobe boundary


def save(volume, path, affine=np.eye(4)):
    nib.save(nib.Nifti1Image(volume, affine), path)


def load(path):
    return np.asarray(nib.load(path).dataobj) > 0


def lobes_meeting_at_the_boundary():
    "the lungs are the slab from x=10 to x=30, with the upper lobe below the boundary and the lower lobe above it"
    lobes = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
    lobes[10:BOUNDARY_X] = LEFT_UPPER
    lobes[BOUNDARY_X:30] = LEFT_LOWER
    return lobes


def ragged_plane_at_the_boundary(rows):
    "a plane on the lobe boundary that jumps by a voxel between columns and has holes, over the given rows"
    rng = np.random.default_rng(0)
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    for y in rows:
        for z in range(5, 35):
            if rng.random() > 0.2:
                mask[BOUNDARY_X + rng.integers(-1, 2), y, z] = True
    return mask


def far_line():
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    mask[FAR_X, 20, 5:35] = True
    return mask


def job_for(tmp_path, views):
    return PostProcessJob(
        nifti=tmp_path / "scan.nii.gz",
        mask_path=tmp_path / "fissures.nii.gz",
        views=views,
        lobes=tmp_path / "lobes.nii.gz",
        out_dir=tmp_path / "post",
    )


def test_a_line_outside_the_lungs_is_removed_and_the_rest_becomes_a_sheet(tmp_path):
    save(lobes_meeting_at_the_boundary(), tmp_path / "lobes.nii.gz")
    plane = ragged_plane_at_the_boundary(range(5, 35))
    save((plane | far_line()).astype(np.uint8), tmp_path / "fissures_axial.nii.gz")
    job = job_for(tmp_path, ["axial"])

    post_process_fissure_masks(job)

    removed, sheet = load(job.removed_lines_path), load(job.sheet_path)
    assert (removed == far_line()).all()
    assert sheet.sum() > 0 and abs(np.nonzero(sheet)[0] - BOUNDARY_X).max() <= 1


def test_the_views_are_cleaned_on_their_own_and_blended_into_one_sheet(tmp_path):
    save(lobes_meeting_at_the_boundary(), tmp_path / "lobes.nii.gz")
    save(
        ragged_plane_at_the_boundary(range(5, 20)).astype(np.uint8),
        tmp_path / "fissures_axial.nii.gz",
    )
    save(
        (ragged_plane_at_the_boundary(range(20, 35)) | far_line()).astype(np.uint8),
        tmp_path / "fissures_coronal.nii.gz",
    )
    job = job_for(tmp_path, ["axial", "coronal"])

    post_process_fissure_masks(job)

    sheet = load(job.sheet_path)
    assert sheet[:, 8:17].any() and sheet[:, 23:32].any()  # both views are in the sheet
    assert (load(job.removed_lines_path) == far_line()).all()


def test_masks_on_another_grid_than_the_lobes_are_refused(tmp_path):
    save(lobes_meeting_at_the_boundary(), tmp_path / "lobes.nii.gz")
    save(
        np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8),
        tmp_path / "fissures_axial.nii.gz",
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    with pytest.raises(ValueError, match="voxel grid of the lobes"):
        post_process_fissure_masks(job_for(tmp_path, ["axial"]))


def test_a_slice_step_of_one_keeps_the_settings_of_the_full_run():
    parameters = sheet_parameters(SheetSettings(), slice_step=1)
    assert parameters == {
        "sigma_pca": 3.0,
        "sigma_along": 3.0,
        "sigma_across": 1.0,
        "min_strength": 0.3,
    }


def test_the_sheet_settings_follow_the_slice_step():
    parameters = sheet_parameters(SheetSettings(), slice_step=3)
    assert parameters["sigma_pca"] == 5.0 and parameters["sigma_along"] == 5.0
    assert parameters["min_strength"] == pytest.approx(0.1)
    assert parameters["sigma_across"] == 1.0  # jitter between slices does not change


def test_a_setting_in_the_config_is_never_changed_by_the_slice_step():
    parameters = sheet_parameters(SheetSettings(min_strength=0.2), slice_step=3)
    assert parameters["min_strength"] == 0.2 and parameters["sigma_along"] == 5.0


def test_a_slice_step_below_one_is_refused(tmp_path):
    with pytest.raises(ValueError):
        PostProcessJob(
            nifti=tmp_path / "scan.nii.gz",
            mask_path=tmp_path / "m.nii.gz",
            lobes=tmp_path / "l.nii.gz",
            out_dir=tmp_path,
            slice_step=0,
        )


class FakeSegmenter:
    "writes the lobes the way a segmenter would, and remembers what it was asked for"

    def __init__(self):
        self.calls = []

    def __call__(self, scan, structures, out_path):
        self.calls.append((scan, list(structures), out_path))
        save(lobes_meeting_at_the_boundary(), out_path)
        return out_path


def test_the_lobes_are_segmented_from_the_scan_when_the_file_does_not_exist(tmp_path):
    save(
        ragged_plane_at_the_boundary(range(5, 35)).astype(np.uint8),
        tmp_path / "fissures_axial.nii.gz",
    )
    job = job_for(tmp_path, ["axial"])
    segmenter = FakeSegmenter()

    post_process_fissure_masks(job, segmenter)

    assert segmenter.calls == [(job.nifti, LUNG_LOBES, job.lobes)]
    assert job.lobes.exists() and load(job.sheet_path).any()


def test_existing_lobes_are_reused_and_the_segmenter_is_not_called(tmp_path):
    save(lobes_meeting_at_the_boundary(), tmp_path / "lobes.nii.gz")
    save(
        ragged_plane_at_the_boundary(range(5, 35)).astype(np.uint8),
        tmp_path / "fissures_axial.nii.gz",
    )
    segmenter = FakeSegmenter()

    post_process_fissure_masks(job_for(tmp_path, ["axial"]), segmenter)

    assert segmenter.calls == []


def test_small_pieces_of_the_sheet_are_dropped_and_saved_separately(tmp_path):
    save(lobes_meeting_at_the_boundary(), tmp_path / "lobes.nii.gz")
    save(
        ragged_plane_at_the_boundary(range(5, 35)).astype(np.uint8),
        tmp_path / "fissures_axial.nii.gz",
    )
    everything = job_for(tmp_path, ["axial"]).model_copy(update={"min_piece_voxels": 0})
    post_process_fissure_masks(everything, FakeSegmenter())
    whole_sheet = load(everything.sheet_path)

    nothing_big_enough = everything.model_copy(
        update={"min_piece_voxels": int(whole_sheet.sum()) + 1}
    )
    post_process_fissure_masks(nothing_big_enough, FakeSegmenter())

    assert not load(nothing_big_enough.sheet_path).any()
    assert (load(nothing_big_enough.removed_pieces_path) == whole_sheet).all()


def test_a_line_inside_the_lungs_is_kept_however_far_it_is_from_the_lobe_boundary(
    tmp_path,
):
    "the lobe boundary is off in severe emphysema, so only the lungs decide"
    save(lobes_meeting_at_the_boundary(), tmp_path / "lobes.nii.gz")
    inside = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    inside[INSIDE_FAR_X, 20, 5:35] = True
    save(
        (ragged_plane_at_the_boundary(range(5, 35)) | inside | far_line()).astype(
            np.uint8
        ),
        tmp_path / "fissures_axial.nii.gz",
    )
    job = job_for(tmp_path, ["axial"])

    post_process_fissure_masks(job, FakeSegmenter())

    removed = load(job.removed_lines_path)
    assert removed[FAR_X, 20, 5:35].all() and not removed[INSIDE_FAR_X].any()
