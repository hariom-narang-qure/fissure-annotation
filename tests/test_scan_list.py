import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np

from fissure_annotation import scan_list
from fissure_annotation.scan_list import (
    ScanFolder,
    ScanListJob,
    annotated_slice_count,
    plan_lines,
    prepare_scan,
    read_series_uids,
    run_scan_list,
    scan_path,
    slices_to_annotate,
)


def write_csv(path: Path, uids: list[str]) -> Path:
    rows = ['series_instance_uid,report'] + [f'{uid},"line one\n\nline two"' for uid in uids]
    path.write_text("\n".join(rows) + "\n")
    return path


def job_for(tmp_path: Path, uids: list[str], **settings) -> ScanListJob:
    return ScanListJob(
        csv=write_csv(tmp_path / "list.csv", uids),
        scans_dir=tmp_path / "scans",
        out_root=tmp_path / "out",
        **settings,
    )


def test_series_uids_are_read_in_order_without_repeats_and_with_multiline_reports(tmp_path):
    csv_path = write_csv(tmp_path / "list.csv", ["b", "a", "b"])
    assert read_series_uids(csv_path, "series_instance_uid") == ["b", "a"]


def test_the_scan_is_in_the_folder_of_the_first_two_characters_of_the_sha1(tmp_path):
    shard = hashlib.sha1(b"1.2.3").hexdigest()[:2]
    assert scan_path(tmp_path, "1.2.3") == tmp_path / shard / "1.2.3.nii.gz"


def test_the_last_slice_is_annotated_even_when_the_step_skips_it():
    assert annotated_slice_count(10, 4) == 4  # 0, 4, 8 and 9
    assert annotated_slice_count(9, 4) == 3  # 0, 4 and 8, which is the last


def test_slices_of_all_views_are_counted_on_the_ras_grid():
    assert slices_to_annotate((10, 9, 5), ["sagittal", "axial"], 4) == 4 + 2  # x has 10 slices, so 0, 4, 8, 9, and z has 5, so 0 and 4


def test_a_scan_with_a_sheet_is_done_and_one_without_is_not(tmp_path):
    folder = ScanFolder(root=tmp_path, uid="a")
    assert not folder.is_done
    folder.sheet_path.parent.mkdir(parents=True)
    folder.sheet_path.touch()
    assert folder.is_done


def test_a_failing_scan_is_recorded_and_the_next_scan_still_runs(tmp_path, monkeypatch):
    ran = []

    def process(job, uid, folder, client):
        ran.append(folder.uid)
        if folder.uid == "a":
            raise RuntimeError("boom")

    monkeypatch.setattr(scan_list, "process_scan", process)
    statuses = run_scan_list(job_for(tmp_path, ["a", "b"]), client=None)
    assert ran == ["a", "b"]
    assert statuses["a"].state == "failed" and "boom" in statuses["a"].error
    assert statuses["b"].state == "done"


def test_a_scan_that_is_done_is_not_run_again(tmp_path, monkeypatch):
    job = job_for(tmp_path, ["a"])
    done = ScanFolder(root=job.out_root, uid="a")
    done.sheet_path.parent.mkdir(parents=True)
    done.sheet_path.touch()
    monkeypatch.setattr(scan_list, "process_scan", lambda *args: 1 / 0)
    assert run_scan_list(job, client=None) == {}


def test_limit_keeps_the_first_scans_of_the_list(tmp_path):
    lines = plan_lines(job_for(tmp_path, ["a", "b", "c"], limit=2))
    assert [line.split()[0] for line in lines[:-1]] == ["a", "b"]


def test_the_plan_names_a_scan_that_is_missing(tmp_path):
    assert "MISSING" in plan_lines(job_for(tmp_path, ["a"]))[0]


def store_scan_with_spacing(job: ScanListJob, uid: str, spacing: tuple[float, float, float]) -> Path:
    path = scan_path(job.scans_dir, uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((6, 6, 5), dtype=np.int16), np.diag([*spacing, 1.0])), path)
    return path


def test_a_scan_with_square_enough_voxels_is_linked_from_the_store(tmp_path):
    job = job_for(tmp_path, ["a"])
    source = store_scan_with_spacing(job, "a", (0.8, 0.8, 1.0))
    folder = ScanFolder(root=job.out_root, uid="a")
    prepare_scan(job, "a", folder)
    assert folder.scan_path.is_symlink() and folder.scan_path.resolve() == source.resolve()


def test_a_scan_with_thick_slices_is_resampled_into_its_folder(tmp_path):
    job = job_for(tmp_path, ["a"])
    store_scan_with_spacing(job, "a", (1.0, 1.0, 3.0))
    folder = ScanFolder(root=job.out_root, uid="a")
    prepare_scan(job, "a", folder)
    image = nib.load(folder.scan_path)
    assert not folder.scan_path.is_symlink()
    assert image.shape == (6, 6, 13)  # 4 gaps of 3 mm become 12 gaps of 1 mm
    assert np.allclose(image.header.get_zooms()[:3], (1.0, 1.0, 1.0))


def test_a_scan_with_slices_thicker_than_the_limit_is_skipped_and_recorded(tmp_path, monkeypatch):
    job = job_for(tmp_path, ["a"])
    store_scan_with_spacing(job, "a", (0.7, 0.7, 10.0))
    statuses = run_scan_list(job, client=None)
    assert statuses["a"].state == "skipped" and "10.0 mm" in statuses["a"].error
    assert "skip" in plan_lines(job)[0]


def test_a_named_run_is_a_subfolder_of_the_scan(tmp_path):
    folder = ScanFolder(root=tmp_path, uid="a", run="flex_step4")
    assert folder.sheet_path == tmp_path / "a" / "flex_step4" / "post" / "sheet.nii.gz"


def test_runs_of_a_scan_list_go_to_their_named_subfolder(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(scan_list, "process_scan", lambda job, uid, folder, client: seen.append(folder.folder))
    run_scan_list(job_for(tmp_path, ["a"], run_name="flex_step4"), client=None)
    assert seen == [tmp_path / "out" / "a" / "flex_step4"]
