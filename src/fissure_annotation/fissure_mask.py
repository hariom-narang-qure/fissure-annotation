"""A NIfTI scan to a NIfTI mask of the fissures the model found, in the same space as the scan.

- Every slice of each chosen view (axial, coronal or sagittal) goes through the waypoint agent, so the cost scales with the number of slices and views.
- The traced lines of each slice are stacked into a 0/1 mask per view, one voxel wide in plane and one slice thick.
- The masks of the views are combined with a voxel-wise OR, so a fissure seen in any view is in the final mask.
- The mask of each view is also saved beside the combined one, as `<mask name>_<view>.nii.gz`.
  - Post-processing judges the lines of each view on its own, which is not possible once the views are combined.
- Slices outside the range, and slices where the model traced nothing, are empty.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Bool
from pydantic import model_validator

from fissure_annotation.agent import AgentSettings
from fissure_annotation.experiment import WaypointExperiment, run_slices
from fissure_annotation.imaging.masks import save_mask_like
from fissure_annotation.imaging.png import read_gray
from fissure_annotation.imaging.slices import View, slice_file_name, write_slices
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.waypoints import RunFolder


class FissureMaskJob(ExperimentSettings):
    nifti: Path
    mask_path: Path
    views: list[View] = ["axial"]
    work_dir: Path | None = (
        None  # one folder per view inside, with its slice PNGs and the agent's points, masks and logs; defaults to a folder next to the mask
    )
    first_slice: int = 0  # slice numbers count along the axis of the view, so a range only makes sense for one view
    last_slice: int | None = None  # the last slice of the view when not given
    workers: int = 16
    agent: AgentSettings = AgentSettings()

    @model_validator(mode="after")
    def slice_range_needs_a_single_view(self):
        if len(self.views) > 1 and (
            self.first_slice != 0 or self.last_slice is not None
        ):
            raise ValueError(
                "first_slice and last_slice count along the axis of a view, so they need exactly one view"
            )
        return self

    @property
    def work_folder(self) -> Path:
        return (
            self.work_dir
            or self.mask_path.parent
            / f"{self.mask_path.name.removesuffix('.nii.gz')}_work"
        )


def view_mask_path(mask_path: Path, view: View) -> Path:
    name = mask_path.name.removesuffix(".nii.gz")
    return mask_path.with_name(f"{name}_{view}.nii.gz")


def view_masks(
    job: FissureMaskJob, view: View, client
) -> dict[int, Bool[np.ndarray, "rows columns"]]:
    folder = job.work_folder / view
    slice_count = write_slices(job.nifti, folder / "slices", view)
    config = WaypointExperiment(
        slices_dir=folder / "slices",
        out_dir=folder / "run",
        first_slice=job.first_slice,
        last_slice=slice_count - 1 if job.last_slice is None else job.last_slice,
        workers=job.workers,
        agent=job.agent.model_copy(update={"view": view}),
    )
    run_slices(config, client)
    run_folder = RunFolder(root=config.out_dir)
    mask_files = {
        z: run_folder.mask_path(config.slices_dir / slice_file_name(z))
        for z in config.slice_indices
    }
    return {z: read_gray(path) > 0 for z, path in mask_files.items() if path.exists()}


def make_fissure_mask(job: FissureMaskJob, client) -> Path:
    masks_by_view = {view: view_masks(job, view, client) for view in job.views}
    for view, masks in masks_by_view.items():
        save_mask_like(job.nifti, {view: masks}, view_mask_path(job.mask_path, view))
    save_mask_like(job.nifti, masks_by_view, job.mask_path)
    return job.mask_path
