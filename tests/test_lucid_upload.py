import json
from pathlib import Path

import pytest

from fissure_annotation import lucid_upload
from fissure_annotation.lucid_upload import (
    LucidUploadJob,
    bundle_series,
    import_statuses,
    require_dispatched,
    s3_root,
    stage_series,
    upload_to_lucid,
)


def job_for(tmp_path: Path, **settings) -> LucidUploadJob:
    (tmp_path / "list.csv").write_text("series_instance_uid\na\nb\n")
    return LucidUploadJob(
        csv=tmp_path / "list.csv",
        out_root=tmp_path / "out",
        project_slug="proj",
        project_name="Project",
        **settings,
    )


def finish_scan(tmp_path: Path, uid: str) -> None:
    "the scan the pipeline used and its sheet, in the folder of the run"
    sheet = tmp_path / "out" / uid / "post" / "sheet.nii.gz"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text("sheet")
    (tmp_path / "out" / uid / "scan.nii.gz").write_text("scan")


def test_the_upload_stops_when_the_bucket_prefix_is_not_set(tmp_path, monkeypatch):
    monkeypatch.delenv("SALVO_BUCKET_PREFIX", raising=False)
    with pytest.raises(ValueError, match="SALVO_BUCKET_PREFIX"):
        s3_root(job_for(tmp_path))


def test_the_s3_root_is_the_bucket_the_prefix_and_the_slug(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me/")
    assert s3_root(job_for(tmp_path)) == "s3://qmuse-annotation/me/proj"


def test_the_bundle_has_the_scan_and_one_mask_finding_per_name(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    series = bundle_series(job_for(tmp_path), "a")
    assert series.scan == "s3://qmuse-annotation/me/proj/a/scan.nii.gz"
    assert [(x.name, x.path) for x in series.annotations] == [
        ("fissure_sheet", "s3://qmuse-annotation/me/proj/a/fissure_sheet.nii.gz")
    ]


def test_staging_links_the_scan_and_each_finding(tmp_path):
    job = job_for(tmp_path)
    finish_scan(tmp_path, "a")
    stage_series(job, "a")
    folder = lucid_upload.stage_dir(job) / "a"
    assert (folder / "scan.nii.gz").read_text() == "scan"
    assert (folder / "fissure_sheet.nii.gz").read_text() == "sheet"


def test_a_dry_run_calls_neither_lucid_nor_s3_and_leaves_out_unfinished_scans(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    commands = []
    monkeypatch.setattr(lucid_upload, "run", lambda command: commands.append(command) or "")
    finish_scan(tmp_path, "a")  # b has no sheet
    left_out = upload_to_lucid(job_for(tmp_path, dry_run=True))
    assert left_out == ["b"]
    assert len(commands) == 1 and commands[0][:3] == ["aws", "s3", "sync"]
    assert "--dryrun" in commands[0]
    bundle = json.loads((tmp_path / "out" / "lucid_bundle_proj.json").read_text())
    assert [series["series_id"] for series in bundle["series"]] == ["a"]


def test_a_bundle_that_was_not_dispatched_is_an_error():
    with pytest.raises(RuntimeError, match="not dispatched"):
        require_dispatched(json.dumps({"dispatch_error": "queue down"}))
    require_dispatched(json.dumps({"dispatch_error": None}))


def test_import_statuses_are_counted():
    output = json.dumps([{"import_status": "ready"}, {"import_status": "ready"}, {"import_status": "failed"}])
    assert import_statuses(output) == {"ready": 2, "failed": 1}


def test_a_series_suffix_makes_its_own_case_and_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    job = job_for(tmp_path, series_suffix=".flex4")
    series = bundle_series(job, "a")
    assert series.series_id == "a.flex4"
    assert series.scan == "s3://qmuse-annotation/me/proj/a.flex4/scan.nii.gz"
    finish_scan(tmp_path, "a")
    stage_series(job, "a")
    assert (lucid_upload.stage_dir(job) / "a.flex4" / "scan.nii.gz").exists()


def test_each_series_is_submitted_on_its_own_and_waited_for(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[:3] == ["salvo-client", "project", "submit-bundle"]:
            return json.dumps({"dispatch_error": None})
        if command[:3] == ["salvo-client", "project", "wait"]:
            return json.dumps([{"import_status": "ready"}])
        return "{}"

    monkeypatch.setattr(lucid_upload, "run", fake_run)
    finish_scan(tmp_path, "a")
    finish_scan(tmp_path, "b")
    upload_to_lucid(job_for(tmp_path))
    salvo_steps = [c[2] for c in commands if c[0] == "salvo-client"]
    assert salvo_steps == ["create", "submit-bundle", "wait", "submit-bundle", "wait"]
