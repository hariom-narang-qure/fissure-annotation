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
    """Answers salvo-client commands like the service.

    After a submission, the case first shows its old status (stale), then importing, then the scripted outcome.
    Each submission of a series takes the next outcome scripted for it.
    """

    def __init__(self, outcomes, stale=None):
        self.outcomes = {uid: list(statuses) for uid, statuses in outcomes.items()}
        self.stale = stale
        self.polls, self.submitted = {}, []

    def __call__(self, command):
        if command[0] != "salvo-client":
            return ""  # aws s3 sync
        action = command[2]
        if action == "submit-bundle":
            self.bundle = json.loads(Path(command[4]).read_text())
            series_id = self.bundle["series"][0]["series_id"]
            self.submitted.append(series_id)
            steps = ([self.stale] if self.stale else []) + ["importing", self.outcomes[series_id].pop(0)]
            self.polls[study_uid("proj", series_id)] = steps
            return json.dumps({"dispatch_error": None})
        if action == "cases":
            rows = []
            for study, steps in self.polls.items():
                rows.append({"study_instance_uid": study, "import_status": steps[0]})
                if len(steps) > 1:
                    steps.pop(0)
            return json.dumps(rows)
        return "{}"


def fast_job(tmp_path, **settings):
    return job_for(tmp_path, **{"retry_wait_seconds": 0, "poll_seconds": 0, "grace_seconds": 0, **settings})


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


def test_a_failed_import_is_retried_under_the_same_id(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    lucid = FakeLucid({"a": ["failed", "failed", "ready"], "b": ["failed"] * 4})
    monkeypatch.setattr(lucid_upload, "run", lucid)
    finish_scan(tmp_path, "a")
    finish_scan(tmp_path, "b")
    report = upload_to_lucid(fast_job(tmp_path))
    assert lucid.submitted == ["a"] * 3 + ["b"] * 4
    assert report.imported == {"a": "a"} and report.failed == ["b"]


def test_the_old_status_of_a_resubmitted_case_is_not_taken_as_the_result(tmp_path, monkeypatch):
    "a case that was ready shows ready until the new import starts, and that import fails here"
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    lucid = FakeLucid({"a": ["failed"]}, stale="ready")
    monkeypatch.setattr(lucid_upload, "run", lucid)
    finish_scan(tmp_path, "a")
    report = upload_to_lucid(fast_job(tmp_path, import_attempts=1, grace_seconds=60))
    assert report.failed == ["a"]


def test_the_report_from_the_csv_is_added_as_a_finding_with_no_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("SALVO_BUCKET_PREFIX", "me")
    (tmp_path / "list.csv").write_text('series_instance_uid,report\na,"EXAM: CT CHEST\n\nIMPRESSION: ok"\n')
    lucid = FakeLucid({"a": ["ready"]})
    monkeypatch.setattr(lucid_upload, "run", lucid)
    finish_scan(tmp_path, "a")
    job = LucidUploadJob(csv=tmp_path / "list.csv", out_root=tmp_path / "out", project_slug="proj", project_name="Project",
                         report_column="report", retry_wait_seconds=0, poll_seconds=0, grace_seconds=0)
    upload_to_lucid(job)
    report_finding = lucid.bundle["series"][0]["annotations"][-1]
    assert report_finding == {"name": "radiologist_report", "report_text": "EXAM: CT CHEST\n\nIMPRESSION: ok"}


def test_a_named_run_is_found_in_its_subfolder(tmp_path):
    job = job_for(tmp_path, run_name="flex_step4")
    folder = tmp_path / "out" / "a" / "flex_step4"
    (folder / "post").mkdir(parents=True)
    (folder / "post" / "sheet.nii.gz").write_text("sheet")
    (folder / "scan.nii.gz").write_text("scan")
    assert lucid_upload.is_ready(job, "a")
