"""A NIfTI scan to a NIfTI mask of the fissures the model found, in the same space as the scan.

- Every slice of the chosen view (axial, coronal or sagittal), or the chosen range, goes through the waypoint agent, so the cost scales with the number of slices.
- The traced lines of each slice are stacked into a 0/1 mask, one voxel wide in plane and one slice thick.
- Slices outside the range, and slices where the model traced nothing, are empty.
"""
from pathlib import Path

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
    view: View = "axial"
    work_dir: Path | None = None  # slice PNGs and the agent's points, masks and logs; defaults to a folder next to the mask
    first_slice: int = 0  # slice numbers count along the axis of the view
    last_slice: int | None = None  # the last slice of the view when not given
    workers: int = 16
    agent: AgentSettings = AgentSettings()

    @property
    def work_folder(self) -> Path:
        return self.work_dir or self.mask_path.parent / f"{self.mask_path.name.removesuffix('.nii.gz')}_{self.view}_work"


def make_fissure_mask(job: FissureMaskJob, client) -> Path:
    slice_count = write_slices(job.nifti, job.work_folder / "slices", job.view)
    config = WaypointExperiment(slices_dir=job.work_folder / "slices", out_dir=job.work_folder / "run", first_slice=job.first_slice,
                                last_slice=slice_count - 1 if job.last_slice is None else job.last_slice, workers=job.workers,
                                agent=job.agent.model_copy(update={"view": job.view}))
    run_slices(config, client)
    folder = RunFolder(root=config.out_dir)
    mask_files = {z: folder.mask_path(config.slices_dir / slice_file_name(z)) for z in config.slice_indices}
    save_mask_like(job.nifti, {z: read_gray(path) > 0 for z, path in mask_files.items() if path.exists()}, job.view, job.mask_path)
    return job.mask_path
