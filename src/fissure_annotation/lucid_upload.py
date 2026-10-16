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
5. The project is created (it is not created again when it exists), the bundle submitted, and the import waited for.

Each finding is a mask with a path, so it shows as an overlay in the viewer.
The viewer did not draw the first of two overlapping masks that shared a case, so give one finding per case when the masks overlap.

The S3 prefix always comes from `$SALVO_BUCKET_PREFIX`, and the upload stops when it is not set.
The commands are run as processes, since neither the AWS CLI nor `salvo-client` is a dependency of this repo.
"""

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from fissure_annotation.scan_list import ScanFolder, read_series_uids
from fissure_annotation.settings import ExperimentSettings

SCAN_NAME = "scan.nii.gz"


class LucidUploadJob(ExperimentSettings):
    csv: Path
    out_root: Path  # where run_scan_list wrote the folder of each scan
    project_slug: str
    project_name: str
    uid_column: str = "series_instance_uid"
    bucket: str = "qmuse-annotation"
    findings: dict[str, str] = {
        "fissure_sheet": "post/sheet.nii.gz"
    }  # the name in the viewer, and the file inside the folder of the scan
    wait_timeout_seconds: int = 600
    limit: int | None = None
    dry_run: bool = False  # stage and list what would be sent, and call neither S3 nor Lucid


class Annotation(BaseModel):
    name: str
    path: str


class BundleSeries(BaseModel):
    series_id: str
    scan: str
    annotations: list[Annotation]


class Bundle(BaseModel):
    series: list[BundleSeries]


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
    folder = ScanFolder(root=job.out_root, uid=uid).folder
    return {name: folder / relative for name, relative in job.findings.items()}


def scan_file(job: LucidUploadJob, uid: str) -> Path:
    return ScanFolder(root=job.out_root, uid=uid).scan_path


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


def stage_series(job: LucidUploadJob, uid: str) -> None:
    folder = stage_dir(job) / uid
    link(scan_file(job, uid), folder / SCAN_NAME)
    for name, path in finding_files(job, uid).items():
        link(path, folder / f"{name}.nii.gz")


def bundle_series(job: LucidUploadJob, uid: str) -> BundleSeries:
    root = f"{s3_root(job)}/{uid}"
    return BundleSeries(
        series_id=uid,
        scan=f"{root}/{SCAN_NAME}",
        annotations=[
            Annotation(name=name, path=f"{root}/{name}.nii.gz")
            for name in job.findings
        ],
    )


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


def wait_for_import(job: LucidUploadJob) -> str:
    return salvo(
        "project",
        "wait",
        job.project_slug,
        "--timeout",
        str(job.wait_timeout_seconds),
    )


def import_statuses(wait_output: str) -> dict[str, int]:
    return dict(Counter(case["import_status"] for case in json.loads(wait_output)))


def upload_to_lucid(job: LucidUploadJob) -> list[str]:
    "returns the series left out because a finding is missing"
    s3_root(job)  # stops here when the prefix is not set
    uids = selected_uids(job)
    ready = [uid for uid in uids if is_ready(job, uid)]
    for uid in ready:
        stage_series(job, uid)
    bundle_path = job.out_root / f"lucid_bundle_{job.project_slug}.json"
    bundle = Bundle(series=[bundle_series(job, uid) for uid in ready])
    bundle_path.write_text(bundle.model_dump_json(indent=2))
    print(sync_to_s3(job))
    if not job.dry_run:
        salvo("project", "create", job.project_slug, job.project_name)
        require_dispatched(
            salvo("project", "submit-bundle", job.project_slug, str(bundle_path))
        )
        print(import_statuses(wait_for_import(job)))
    return [uid for uid in uids if uid not in ready]
