import json
from pathlib import Path

import pytest

from fissure_annotation import lucid_upload
from fissure_annotation.lucid_upload import (
    LucidUploadJob,
    bundle_series,
    study_uid,
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
    report = upload_to_lucid(job_for(tmp_path, dry_run=True))
    assert report.left_out == ["b"] and report.imported == {}
    assert len(commands) == 1 and commands[0][:3] == ["aws", "s3", "sync"]
    assert "--dryrun" in commands[0]
    bundle = json.loads((tmp_path / "out" / "lucid_bundle_proj.json").read_text())
    assert [series["series_id"] for series in bundle["series"]] == ["a"]


def test_a_bundle_that_was_not_dispatched_is_an_error():
    with pytest.raises(RuntimeError, match="not dispatched"):
        require_dispatched(json.dumps({"dispatch_error": "queue down"}))
    require_dispatched(json.dumps({"dispatch_error": None}))


def test_a_series_suffix_makes_its_own_case_and_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    job = job_for(tmp_path, series_suffix=".flex4")
    series = bundle_series(job, "a")
    assert series.series_id == "a.flex4"
    assert series.scan == "s3://qmuse-annotation/me/proj/a.flex4/scan.nii.gz"
    finish_scan(tmp_path, "a")
    stage_series(job, "a")
    assert (lucid_upload.stage_dir(job) / "a.flex4" / "scan.nii.gz").exists()


class FakeLucid:
    "answers salvo-client commands; each submitted series id imports with the next scripted status for its scan"

    def __init__(self, outcomes):
        self.outcomes = {uid: list(statuses) for uid, statuses in outcomes.items()}
        self.cases, self.submitted = {}, []

    def __call__(self, command):
        if command[0] != "salvo-client":
            return ""  # aws s3 sync
        action = command[2]
        if action == "submit-bundle":
            series_id = json.loads(Path(command[4]).read_text())["series"][0]["series_id"]
            self.submitted.append(series_id)
            uid = series_id.split(".retry")[0]
            self.cases[study_uid("proj", series_id)] = self.outcomes[uid].pop(0)
            return json.dumps({"dispatch_error": None})
        if action == "cases":
            return json.dumps([{"study_instance_uid": k, "import_status": v} for k, v in self.cases.items()])
        return "{}"


def fast_job(tmp_path, **settings):
    return job_for(tmp_path, retry_wait_seconds=0, poll_seconds=0, **settings)


def test_the_study_uid_is_minted_as_the_lucid_service_mints_it():
    "a real case: Lucid listed this series of project fissure-step-compare under this study uid"
    series = "1.2.392.200036.9116.2.6.1.54365.1690017023.1783390643.994211"
    assert study_uid("fissure-step-compare", series) == "2.25.92264679872701650509228289789433897723"


def test_each_series_is_submitted_on_its_own_and_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    lucid = FakeLucid({"a": ["ready"], "b": ["ready"]})
    monkeypatch.setattr(lucid_upload, "run", lucid)
    finish_scan(tmp_path, "a")
    finish_scan(tmp_path, "b")
    report = upload_to_lucid(fast_job(tmp_path))
    assert lucid.submitted == ["a", "b"]
    assert report.imported == {"a": "a", "b": "b"} and report.failed == []


def test_a_failed_import_is_retried_under_a_fresh_id(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    lucid = FakeLucid({"a": ["failed", "failed", "ready"], "b": ["failed"] * 4})
    monkeypatch.setattr(lucid_upload, "run", lucid)
    finish_scan(tmp_path, "a")
    finish_scan(tmp_path, "b")
    report = upload_to_lucid(fast_job(tmp_path))
    assert lucid.submitted[:3] == ["a", "a.retry1", "a.retry2"]
    assert report.imported == {"a": "a.retry2"}
    assert report.failed == ["b"] and lucid.submitted[3:] == ["b", "b.retry1", "b.retry2", "b.retry3"]
