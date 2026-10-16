"""Run the whole fissure pipeline over a list of scans, one scan after the other.

Why:
- Judging the agent on one scan says little about it, so it is run on a list of series from a CSV.
- Every scan is independent, so one failing scan must not stop the others.

How:
1. The series UIDs are read from a column of the CSV, in order and without repeats.
2. The scan of a series is found with `scan_path`, the layout of the qmuse scan store.
   - A scan with slices thicker than `max_slice_mm` is skipped, since fissures are lost between such slices.
   - A scan whose slices are more than `max_slice_to_pixel_ratio` times its pixels is resampled along z (`imaging/resample.py`).
   - The scan the pipeline uses is kept in the folder of the scan, as a link to the store when it needs no resampling.
3. For each scan, the agent is run on every `slice_step`-th slice of each view (`make_fissure_mask`).
4. Then the masks are cleaned and blended into one sheet (`post_process_fissure_masks`).
5. Each scan gets its own folder under `out_root`, see `ScanFolder`.
   - A scan with a sheet in its folder is done, and is skipped on the next run.
   - A run that stopped halfway is finished by running it again, since the agent skips slices that have a log.
   - A scan that raised an error, or was skipped, is recorded in its `status.json` and the list goes on.

The agent costs about the same per slice, so `estimated_cost_usd` is the number of slices times a billed price per slice.
That price was measured on full runs of gpt-6.1-sol at a step of 3 (see COST_SAVING.md), so it is a rough check before a run.
"""

import csv
import hashlib
import time
import traceback
from pathlib import Path
from typing import Literal

import nibabel as nib
from pydantic import BaseModel, Field

from fissure_annotation.agent import AgentSettings
from fissure_annotation.fissure_mask import FissureMaskJob, make_fissure_mask
from fissure_annotation.imaging.resample import (
    resample_z_to_pixel_spacing,
    resampled_slice_count,
    slice_to_pixel_ratio,
)
from fissure_annotation.imaging.slices import VIEW_AXIS, View
from fissure_annotation.postprocess import (
    PostProcessJob,
    SheetSettings,
    post_process_fissure_masks,
)
from fissure_annotation.segmentation import TotalSegmentatorSettings
from fissure_annotation.settings import ExperimentSettings

BILLED_USD_PER_SLICE = (0.022, 0.026)


class ScanListJob(ExperimentSettings):
    csv: Path
    scans_dir: Path  # the qmuse scan store, one folder per first two characters of the sha1 of the UID
    out_root: Path
    uid_column: str = "series_instance_uid"
    views: list[View] = ["axial", "coronal", "sagittal"]
    slice_step: int = Field(
        default=4, ge=1
    )  # the agent annotates every slice_step-th slice of each view, and post-processing follows it
    workers: int = 16  # slices in flight at once inside one scan
    max_slice_mm: float = 3.0  # thicker slices are skipped
    max_slice_to_pixel_ratio: float = 1.5  # above it the scan is resampled along z to its pixel spacing
    limit: int | None = None  # only the first scans of the list, to try the run on a few
    dry_run: bool = False  # the script prints the plan and calls nothing
    agent: AgentSettings = AgentSettings()
    sheet: SheetSettings = SheetSettings()
    totalsegmentator: TotalSegmentatorSettings = TotalSegmentatorSettings()


class ScanFolder(BaseModel):
    root: Path
    uid: str

    @property
    def folder(self) -> Path:
        return self.root / self.uid

    @property
    def scan_path(self) -> Path:
        return self.folder / "scan.nii.gz"

    @property
    def mask_path(self) -> Path:
        return self.folder / "fissures.nii.gz"

    @property
    def lobes_path(self) -> Path:
        return self.folder / "lobes.nii.gz"

    @property
    def post_dir(self) -> Path:
        return self.folder / "post"

    @property
    def sheet_path(self) -> Path:
        return self.post_dir / "sheet.nii.gz"

    @property
    def status_path(self) -> Path:
        return self.folder / "status.json"

    @property
    def is_done(self) -> bool:
        return self.sheet_path.exists()


class ScanStatus(BaseModel):
    state: Literal["done", "failed", "skipped"]
    seconds: float
    error: str | None = None


def read_series_uids(csv_path: Path, column: str) -> list[str]:
    "the CSV has long multi-line reports, so it is read with the csv module and not by lines"
    with csv_path.open(newline="") as file:
        uids = [row[column].strip() for row in csv.DictReader(file)]
    return list(dict.fromkeys(uid for uid in uids if uid))


def scan_path(scans_dir: Path, series_uid: str) -> Path:
    shard = hashlib.sha1(series_uid.encode()).hexdigest()[:2]
    return scans_dir / shard / f"{series_uid}.nii.gz"


class ScanSkipped(Exception):
    pass


def store_scan(job: ScanListJob, uid: str) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(scan_path(job.scans_dir, uid)))


def needs_resampling(job: ScanListJob, image: nib.Nifti1Image) -> bool:
    return slice_to_pixel_ratio(image) > job.max_slice_to_pixel_ratio


def check_slice_thickness(job: ScanListJob, image: nib.Nifti1Image) -> None:
    slice_mm = float(image.header.get_zooms()[2])
    if slice_mm > job.max_slice_mm:
        raise ScanSkipped(f"slices of {slice_mm} mm, thicker than {job.max_slice_mm} mm")


def prepare_scan(job: ScanListJob, uid: str, folder: ScanFolder) -> Path:
    "a resampled scan that exists already is reused, a link is made again in case it pointed elsewhere"
    image = store_scan(job, uid)
    check_slice_thickness(job, image)
    resample = needs_resampling(job, image)
    if resample and folder.scan_path.exists() and not folder.scan_path.is_symlink():
        return folder.scan_path
    folder.folder.mkdir(parents=True, exist_ok=True)
    folder.scan_path.unlink(missing_ok=True)
    if resample:
        resample_z_to_pixel_spacing(scan_path(job.scans_dir, uid), folder.scan_path)
    else:
        folder.scan_path.symlink_to(scan_path(job.scans_dir, uid))
    return folder.scan_path


def uids_of(job: ScanListJob) -> list[str]:
    uids = read_series_uids(job.csv, job.uid_column)
    return uids if job.limit is None else uids[: job.limit]


def annotated_slice_count(length: int, step: int) -> int:
    "every step-th slice from the first, and always the last, as the agent run does"
    return len({*range(0, length, step), length - 1})


def prepared_shape(job: ScanListJob, image: nib.Nifti1Image) -> tuple[int, int, int]:
    "the shape of the scan the pipeline will use, without resampling it"
    x, y, z = image.shape[:3]
    if needs_resampling(job, image):
        z = resampled_slice_count(z, slice_to_pixel_ratio(image))
    return x, y, z


def slices_to_annotate(shape: tuple[int, int, int], views: list[View], step: int) -> int:
    "the shape is in RAS order, which is the order the slices of a view count in"
    return sum(annotated_slice_count(shape[VIEW_AXIS[view]], step) for view in views)


def estimated_cost_usd(slice_count: int) -> tuple[float, float]:
    low, high = BILLED_USD_PER_SLICE
    return slice_count * low, slice_count * high


def mask_job_for(job: ScanListJob, scan: Path, folder: ScanFolder) -> FissureMaskJob:
    return FissureMaskJob(
        nifti=scan,
        mask_path=folder.mask_path,
        views=job.views,
        slice_step=job.slice_step,
        workers=job.workers,
        agent=job.agent,
    )


def post_job_for(job: ScanListJob, scan: Path, folder: ScanFolder) -> PostProcessJob:
    return PostProcessJob(
        nifti=scan,
        mask_path=folder.mask_path,
        views=job.views,
        lobes=folder.lobes_path,
        out_dir=folder.post_dir,
        slice_step=job.slice_step,
        sheet=job.sheet,
        totalsegmentator=job.totalsegmentator,
    )


def process_scan(job: ScanListJob, uid: str, folder: ScanFolder, client) -> None:
    scan = prepare_scan(job, uid, folder)
    make_fissure_mask(mask_job_for(job, scan, folder), client)
    post_process_fissure_masks(post_job_for(job, scan, folder))


def record_status(folder: ScanFolder, status: ScanStatus) -> None:
    folder.folder.mkdir(parents=True, exist_ok=True)
    folder.status_path.write_text(status.model_dump_json(indent=2))


def run_one(job: ScanListJob, uid: str, client) -> ScanStatus:
    "an error is recorded and returned, so the next scan still runs"
    folder = ScanFolder(root=job.out_root, uid=uid)
    started = time.monotonic()
    try:
        process_scan(job, uid, folder, client)
        status = ScanStatus(state="done", seconds=time.monotonic() - started)
    except ScanSkipped as reason:
        status = ScanStatus(
            state="skipped", seconds=time.monotonic() - started, error=str(reason)
        )
    except Exception:
        status = ScanStatus(
            state="failed",
            seconds=time.monotonic() - started,
            error=traceback.format_exc(),
        )
    record_status(folder, status)
    return status


def run_scan_list(job: ScanListJob, client) -> dict[str, ScanStatus]:
    statuses: dict[str, ScanStatus] = {}
    uids = uids_of(job)
    for number, uid in enumerate(uids, start=1):
        if ScanFolder(root=job.out_root, uid=uid).is_done:
            print(f"[{number}/{len(uids)}] {uid} done already", flush=True)
            continue
        statuses[uid] = run_one(job, uid, client)
        print(f"[{number}/{len(uids)}] {uid} {statuses[uid].state}", flush=True)
    return statuses


def planned_slices(job: ScanListJob, uid: str) -> int:
    image = store_scan(job, uid)
    if float(image.header.get_zooms()[2]) > job.max_slice_mm:
        return 0
    return slices_to_annotate(prepared_shape(job, image), job.views, job.slice_step)


def plan_line(job: ScanListJob, uid: str) -> str:
    image = store_scan(job, uid)
    slice_mm = float(image.header.get_zooms()[2])
    if slice_mm > job.max_slice_mm:
        return f"{uid} skip, slices of {slice_mm} mm"
    note = ", resampled along z" if needs_resampling(job, image) else ""
    return f"{uid} {planned_slices(job, uid)} slices{note}"


def plan_lines(job: ScanListJob) -> list[str]:
    "one line per scan, then the total, without calling the model"
    lines, total_slices = [], 0
    for uid in uids_of(job):
        scan = scan_path(job.scans_dir, uid)
        if ScanFolder(root=job.out_root, uid=uid).is_done:
            lines.append(f"{uid} done already")
        elif not scan.exists():
            lines.append(f"{uid} MISSING {scan}")
        else:
            lines.append(plan_line(job, uid))
            total_slices += planned_slices(job, uid)
    low, high = estimated_cost_usd(total_slices)
    lines.append(
        f"total {total_slices} slices, about ${low:.0f} to ${high:.0f} billed"
    )
    return lines
