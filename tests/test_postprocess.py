import nibabel as nib
import numpy as np
import pytest

from fissure_annotation.island_rules import DropIslandsUnder, DropSmallIslands
from fissure_annotation.postprocess import (
    PostProcessJob,
    PostProcessReport,
    SheetSettings,
    post_process_fissure_masks,
    sheet_parameters,
)

SIZE = 40
PLANE_X = 20


def save(volume, path, affine=np.eye(4)):
    nib.save(nib.Nifti1Image(volume, affine), path)


def load(path):
    return np.asarray(nib.load(path).dataobj) > 0


def read_report(job):
    return PostProcessReport.model_validate_json(job.report_path.read_text())


def removed_by(job, step):
    labels = np.asarray(nib.load(job.removed_path).dataobj)
    return labels == read_report(job).label_of(step)


def ragged_plane(rows):
    "a plane at PLANE_X that jumps by a voxel between columns and has holes, over the given rows"
    rng = np.random.default_rng(0)
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    for y in rows:
        for z in range(5, 35):
            if rng.random() > 0.2:
                mask[PLANE_X + rng.integers(-1, 2), y, z] = True
    return mask


def job_for(tmp_path, views, **settings):
    return PostProcessJob(
        mask_path=tmp_path / "fissures.nii.gz",
        views=views,
        out_dir=tmp_path / "post",
        **settings,
    )


def test_a_ragged_plane_becomes_a_thin_sheet(tmp_path):
    save(ragged_plane(range(5, 35)).astype(np.uint8), tmp_path / "fissures_axial.nii.gz")
    job = job_for(tmp_path, ["axial"])

    post_process_fissure_masks(job)

    sheet = load(job.sheet_path)
    assert sheet.sum() > 0 and abs(np.nonzero(sheet)[0] - PLANE_X).max() <= 1


def test_the_views_are_blended_into_one_sheet(tmp_path):
    save(ragged_plane(range(5, 20)).astype(np.uint8), tmp_path / "fissures_axial.nii.gz")
    save(ragged_plane(range(20, 35)).astype(np.uint8), tmp_path / "fissures_coronal.nii.gz")
    job = job_for(tmp_path, ["axial", "coronal"])

    post_process_fissure_masks(job)

    sheet = load(job.sheet_path)
    assert sheet[:, 8:17].any() and sheet[:, 23:32].any()


def test_views_on_different_grids_are_refused(tmp_path):
    save(np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8), tmp_path / "fissures_axial.nii.gz")
    save(
        np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8),
        tmp_path / "fissures_coronal.nii.gz",
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    with pytest.raises(ValueError, match="voxel grid of the first view"):
        post_process_fissure_masks(job_for(tmp_path, ["axial", "coronal"]))


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
        job_for(tmp_path, ["axial"], slice_step=0)


def test_the_island_rules_trim_the_sheet_and_write_a_report(tmp_path):
    save(ragged_plane(range(5, 35)).astype(np.uint8), tmp_path / "fissures_axial.nii.gz")
    everything = job_for(tmp_path, ["axial"], island_rules=[])
    post_process_fissure_masks(everything)
    whole_sheet = load(everything.sheet_path)

    drop_all_but_largest = everything.model_copy(
        update={"island_rules": [DropSmallIslands(min_share_of_largest=1.0)]}
    )
    post_process_fissure_masks(drop_all_but_largest)

    kept = load(drop_all_but_largest.sheet_path)
    removed = removed_by(drop_all_but_largest, "drop_small_islands")
    assert ((kept | removed) == whole_sheet).all() and not (kept & removed).any()
    assert read_report(drop_all_but_largest).islands.island_voxels_after == [int(kept.sum())]


def test_islands_under_the_voxel_floor_are_removed_under_their_rule(tmp_path):
    save(ragged_plane(range(5, 35)).astype(np.uint8), tmp_path / "fissures_axial.nii.gz")
    everything = job_for(tmp_path, ["axial"], island_rules=[])
    post_process_fissure_masks(everything)
    whole_sheet = load(everything.sheet_path)

    nothing_big_enough = everything.model_copy(
        update={"island_rules": [DropIslandsUnder(min_voxels=int(whole_sheet.sum()) + 1)]}
    )
    post_process_fissure_masks(nothing_big_enough)

    assert not load(nothing_big_enough.sheet_path).any()
    assert (removed_by(nothing_big_enough, "drop_islands_under") == whole_sheet).all()


def test_the_report_names_a_label_for_every_rule_in_the_order_they_ran(tmp_path):
    save(ragged_plane(range(5, 35)).astype(np.uint8), tmp_path / "fissures_axial.nii.gz")
    job = job_for(tmp_path, ["axial"], island_rules=[DropIslandsUnder(), DropSmallIslands()])
    post_process_fissure_masks(job)
    legend = [(removal.label, removal.step) for removal in read_report(job).removed]
    assert legend == [(1, "drop_islands_under"), (2, "drop_small_islands")]
