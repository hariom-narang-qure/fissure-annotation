"""Upload the results of a scan list to a Lucid project, so they can be looked at in the viewer.

Why:
- The viewer reads scans and masks from S3, and the project is filled from a bundle that lists them.

How:
1. Series whose folder has the scan and every wanted file (the findings) are picked, and the others are reported and left out.
   - The scan in the folder is the one the pipeline used, which may be resampled, so the masks are on its grid.
2. A staging folder gets one folder per series, with symlinks to the scan as `scan.nii.gz` and to each finding as `<name>.nii.gz`.
3. `aws s3 sync` copies the staging folder to `s3://<bucket>/<$SALVO_BUCKET_PREFIX>/<project slug>/`.
   - A file that is there already with the same size is not sent again, so a second run finishes a partial upload.
4. A `bundle.json` lists, for each series, the S3 scan and one mask finding per name, in the format of `salvo-client`.
   - With `report_column` set, the report of the series in that CSV column is added as a finding with no mask.
   - Lucid shows such a finding's `report_text` in the Candidate Review panel, so the report can be read next to the scan.
5. The project is created (it is not created again when it exists).
6. The series are submitted one at a time, and each import is polled until its own case is ready or failed.
   - Submitting 13 cases at once made the ingest service time out on 7 of them (ReadTimeout).
   - The service stores the whole CT series in Orthanc in one request with a 30 s timeout, so a busy service fails imports.
   - A resubmitted case keeps its old status until the import starts, so a status counts only once the import was seen
     running, or after `grace_seconds`. `project wait` reads the old status, so it is not used.
7. A failed import is submitted again under the same id, after `retry_wait_seconds`.
   - Waiting is what made failed imports go through, 5 to 10 minutes later.
   - The same id updates the case: the import is idempotent, and new findings are added to it.
     - So submitting an uploaded series again with a report adds the report to the case already there.
   - A failed import keeps the case's findings, but marks it failed until an import succeeds.

Each finding is a mask with a path, so it shows as an overlay in the viewer.
The viewer did not draw the first of two overlapping masks that shared a case, so give one finding per case when the masks overlap.

The S3 prefix always comes from `$SALVO_BUCKET_PREFIX`, and the upload stops when it is not set.
The commands are run as processes, since neither the AWS CLI nor `salvo-client` is a dependency of this repo.
"""

import csv
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from pydantic import BaseModel

from fissure_annotation.scan_list import ScanFolder, read_series_uids
from fissure_annotation.settings import ExperimentSettings

SCAN_NAME = "scan.nii.gz"


class LucidUploadJob(ExperimentSettings):
    csv: Path
    out_root: Path  # where run_scan_list wrote the folder of each scan
    run_name: str = ""  # the run's subfolder in each scan folder, as in the scan list config
    project_slug: str
    project_name: str
    uid_column: str = "series_instance_uid"
    bucket: str = "qmuse-annotation"
    findings: dict[str, str] = {
        "fissure_sheet": "post/sheet.nii.gz"
    }  # the name in the viewer, and the file inside the folder of the scan
    series_suffix: str = ""  # added to each series id, so a second upload of a scan becomes its own case and does not merge into an earlier one
    wait_timeout_seconds: int = 1500  # one import is given up as stuck after this long
    import_attempts: int = 4  # the first submission and up to three retries under fresh ids
    retry_wait_seconds: float = 300.0  # the wait after a failed import, before the next attempt
    poll_seconds: float = 15.0
    grace_seconds: float = 60.0  # a resubmitted case shows its old status until the import starts
    report_column: str | None = None  # a CSV column with the report of each series, added as a finding with no mask
    report_finding: str = "radiologist_report"
    limit: int | None = None
    dry_run: bool = False  # stage and list what would be sent, and call neither S3 nor Lucid


class Annotation(BaseModel):
    "a mask finding has a path; a finding with only report_text is shown as text, for the radiologist to draw"

    name: str
    path: str | None = None
    report_text: str | None = None


class BundleSeries(BaseModel):
    series_id: str
    scan: str
    annotations: list[Annotation]


class Bundle(BaseModel):
    series: list[BundleSeries]


class UploadReport(BaseModel):
    left_out: list[str]  # series uids without every finding, so not uploaded
    imported: dict[str, str]  # series uid to the series id of its case in Lucid
    failed: list[str]  # series uids whose import failed on every attempt


def bucket_prefix() -> str:
    prefix = os.environ.get("SALVO_BUCKET_PREFIX")
    if not prefix:
        raise ValueError(
            "SALVO_BUCKET_PREFIX is not set, and no default prefix is assumed"
        )
    return prefix.strip("/")


def s3_root(job: LucidUploadJob) -> str:
    return f"s3://{job.bucket}/{bucket_prefix()}/{job.project_slug}"


def stage_dir(job: LucidUploadJob) -> Path:
    return job.out_root / "lucid_stage" / job.project_slug


def finding_files(job: LucidUploadJob, uid: str) -> dict[str, Path]:
    folder = ScanFolder(root=job.out_root, uid=uid, run=job.run_name).folder
    return {name: folder / relative for name, relative in job.findings.items()}


def scan_file(job: LucidUploadJob, uid: str) -> Path:
    return ScanFolder(root=job.out_root, uid=uid, run=job.run_name).scan_path


def is_ready(job: LucidUploadJob, uid: str) -> bool:
    files = [scan_file(job, uid), *finding_files(job, uid).values()]
    return all(path.exists() for path in files)


def selected_uids(job: LucidUploadJob) -> list[str]:
    uids = read_series_uids(job.csv, job.uid_column)
    return uids if job.limit is None else uids[: job.limit]


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.unlink(missing_ok=True)
    link_path.symlink_to(target.resolve())


def series_id(job: LucidUploadJob, uid: str) -> str:
    return uid + job.series_suffix


def stage_series(job: LucidUploadJob, uid: str) -> None:
    folder = stage_dir(job) / series_id(job, uid)
    link(scan_file(job, uid), folder / SCAN_NAME)
    for name, path in finding_files(job, uid).items():
        link(path, folder / f"{name}.nii.gz")


def reports_of(job: LucidUploadJob) -> dict[str, str]:
    "the CSV has long multi-line reports, so it is read with the csv module"
    if job.report_column is None:
        return {}
    with job.csv.open(newline="") as file:
        return {row[job.uid_column].strip(): row[job.report_column].strip() for row in csv.DictReader(file)}


def bundle_series(job: LucidUploadJob, uid: str, report: str | None = None) -> BundleSeries:
    root = f"{s3_root(job)}/{series_id(job, uid)}"
    annotations = [Annotation(name=name, path=f"{root}/{name}.nii.gz") for name in job.findings]
    if report:
        annotations.append(Annotation(name=job.report_finding, report_text=report))
    return BundleSeries(series_id=series_id(job, uid), scan=f"{root}/{SCAN_NAME}", annotations=annotations)


def sync_to_s3(job: LucidUploadJob) -> str:
    "--follow-symlinks sends the files the links point to"
    command = ["aws", "s3", "sync", str(stage_dir(job)), s3_root(job), "--follow-symlinks"]
    if job.dry_run:
        command.append("--dryrun")
    return run(command)


def salvo(*arguments: str) -> str:
    return run(["salvo-client", *arguments])


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def require_dispatched(submit_output: str) -> None:
    "a bundle that was accepted but not dispatched has imported nothing"
    error = json.loads(submit_output).get("dispatch_error")
    if error:
        raise RuntimeError(f"the bundle was not dispatched: {error}")


# The namespace and format of `nifti_dicom.mint_study_uid` in the salvo workflow service, copied as salvo's own upload script does.
# The service mints the study uid from (project, series id), and `project cases` lists cases by that uid.
STUDY_UID_NAMESPACE = uuid.UUID("b0c88b17-bb8c-49fb-b16a-7ff6de7a1978")
UNFINISHED_IMPORT = {None, "pending", "importing"}  # None: the case is not listed yet


def study_uid(project_slug: str, series_id: str) -> str:
    value = uuid.uuid5(STUDY_UID_NAMESPACE, "|".join(["study", project_slug, series_id]))
    return f"2.25.{value.int}"


def import_status(job: LucidUploadJob, study: str) -> str | None:
    cases = json.loads(salvo("project", "cases", job.project_slug))
    return next((case["import_status"] for case in cases if case["study_instance_uid"] == study), None)


def submit(job: LucidUploadJob, series: BundleSeries) -> None:
    path = job.out_root / "lucid_bundle_one.json"
    path.write_text(Bundle(series=[series]).model_dump_json(indent=2, exclude_none=True))
    require_dispatched(salvo("project", "submit-bundle", job.project_slug, str(path)))


def wait_for_case(job: LucidUploadJob, study: str) -> str | None:
    "polls until the case is ready or failed; a stuck import comes back as its last status"
    started = time.monotonic()
    status, seen_running = None, False
    while time.monotonic() - started < job.wait_timeout_seconds:
        time.sleep(job.poll_seconds)
        status = import_status(job, study)
        seen_running = seen_running or status in UNFINISHED_IMPORT
        settled = seen_running or time.monotonic() - started >= job.grace_seconds
        if status not in UNFINISHED_IMPORT and settled:
            break
    return status


def import_with_retries(job: LucidUploadJob, series: BundleSeries) -> bool:
    for attempt in range(job.import_attempts):
        if attempt > 0:
            time.sleep(job.retry_wait_seconds)
        submit(job, series)
        status = wait_for_case(job, study_uid(job.project_slug, series.series_id))
        print(series.series_id, f"attempt {attempt + 1}:", status, flush=True)
        if status == "ready":
            return True
    return False


def upload_to_lucid(job: LucidUploadJob) -> UploadReport:
    s3_root(job)  # stops here when the prefix is not set
    uids = selected_uids(job)
    ready = [uid for uid in uids if is_ready(job, uid)]
    for uid in ready:
        stage_series(job, uid)
    bundle_path = job.out_root / f"lucid_bundle_{job.project_slug}.json"
    reports = reports_of(job)
    bundle = Bundle(series=[bundle_series(job, uid, reports.get(uid)) for uid in ready])
    bundle_path.write_text(bundle.model_dump_json(indent=2, exclude_none=True))
    print(sync_to_s3(job))
    report = UploadReport(left_out=[uid for uid in uids if uid not in ready], imported={}, failed=[])
    if job.dry_run:
        return report
    salvo("project", "create", job.project_slug, job.project_name)
    for uid, series in zip(ready, bundle.series):
        if import_with_retries(job, series):
            report.imported[uid] = series.series_id
        else:
            report.failed.append(uid)
    return report
