"""Blend the per-view fissure masks into one thin sheet, and trim it with the island rules.

Why:
- The masks from `make_fissure_mask` come from lines traced slice by slice, so they have false lines and look ragged in 3D.
  - False lines are traces of the body outline, the table, the diaphragm and the hilum.
  - Ragged means a stack of separate lines with holes between them, most of all where a fissure lies almost flat in the slice plane.
- The sheet is cut into islands, and all trimming is done by rules on the islands, so one report says what each rule did.

How:
1. The masks of the views are OR-ed into one mask.
2. The union is blurred along the sheet and thinned to one voxel (`imaging/sheet.py`).
3. The island rules trim the sheet, or flag the scan for review (`island_rules.py`).
4. Everything removed is saved as one label map, `removed.nii.gz`, with a label per rule (`removals.py`).
   - `report.json` has the legend of the labels, the island sizes and the review flags.
   - Both sit in the scan's output folder, next to `sheet.nii.gz`.

Annotating every `slice_step`-th slice cuts the cost of the model by about that factor.
- Fewer slices give less density on a sheet and longer holes between the lines.
- So the sheet settings that the config leaves out follow the step (`sheet_parameters`):
  - The density cutoff is divided by the step.
  - The neighbourhood and the blur along the sheet grow by one voxel per extra step.
- The rule was found on one scan by comparing the sheet with the sheet of the full run.
  - At a step of 3 it matched it, with recall 0.99 and precision 1.0 within 2.4 mm.
  - The settings of the full run lost 11% to 13% of the sheet at that step.
- Settings that the config sets are never changed.

An earlier version removed lines mostly outside the lungs of a TotalSegmentator lobe segmentation, before blending.
- It was dropped, since TotalSegmentator is not trusted on severe scans.
- It did remove real junk: on 53 earlier runs, without it 24 sheets gained traces of the body outline and the table.
  - Those traces form large islands, of up to 13k voxels, that the size rules keep.
  - So those scans are flagged by the extra-island rules for now, and a better rule for them is open.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Bool, UInt8
from pydantic import BaseModel, Field

from fissure_annotation.fissure_mask import view_mask_path
from fissure_annotation.imaging.sheet import (
    DEFAULT_MIN_STRENGTH,
    DEFAULT_SIGMA_ACROSS_VOXELS,
    DEFAULT_SIGMA_ALONG_VOXELS,
    DEFAULT_SIGMA_PCA_VOXELS,
    oriented_sheet,
)
from fissure_annotation.imaging.slices import View
from fissure_annotation.island_rules import (
    IslandReport,
    IslandRule,
    apply_island_rules,
    default_island_rules,
)
from fissure_annotation.removals import RemovalStep, RemovedVoxels
from fissure_annotation.settings import ExperimentSettings

VoxelMask = Bool[np.ndarray, "x y z"]


class SheetSettings(ExperimentSettings):
    "a setting left out takes the value for the slice step of the job, see sheet_parameters"

    sigma_pca: float | None = (
        None  # voxels, the size of the neighbourhood that decides the sheet direction
    )
    sigma_along: float | None = (
        None  # voxels, the blur along the sheet, which sets the largest hole that is filled
    )
    sigma_across: float | None = (
        None  # voxels, the blur across the sheet, which sets how much jitter is averaged
    )
    min_strength: float | None = (
        None  # of the density of a full one voxel plane, lines and specks stay under it
    )


class PostProcessReport(BaseModel):
    "everything the post-process did to one scan, saved as report.json in the scan's folder"

    removed: list[RemovalStep]  # the legend of removed.nii.gz
    islands: IslandReport

    @property
    def needs_review(self) -> bool:
        return self.islands.needs_review

    def label_of(self, step: str) -> int:
        return next(removal.label for removal in self.removed if removal.step == step)


class PostProcessJob(ExperimentSettings):
    mask_path: Path  # the combined mask of make_fissure_mask, the mask of each view is found next to it
    views: list[View] = ["axial"]
    out_dir: Path
    slice_step: int = Field(
        default=1, ge=1
    )  # the model annotated every slice_step-th slice of each view, 2 for every other slice
    sheet: SheetSettings = SheetSettings()
    island_rules: list[IslandRule] = Field(default_factory=default_island_rules)  # run in this order, see island_rules.py

    @property
    def sheet_path(self) -> Path:
        return self.out_dir / "sheet.nii.gz"

    @property
    def removed_path(self) -> Path:
        return self.out_dir / "removed.nii.gz"

    @property
    def report_path(self) -> Path:
        return self.out_dir / "report.json"


def sheet_parameters(settings: SheetSettings, slice_step: int) -> dict[str, float]:
    for_step = {
        "sigma_pca": DEFAULT_SIGMA_PCA_VOXELS + slice_step - 1,
        "sigma_along": DEFAULT_SIGMA_ALONG_VOXELS + slice_step - 1,
        "sigma_across": DEFAULT_SIGMA_ACROSS_VOXELS,  # jitter between slices does not depend on how many are annotated
        "min_strength": DEFAULT_MIN_STRENGTH / slice_step,
    }
    return {
        name: for_step[name] if value is None else value
        for name, value in settings.model_dump().items()
    }


def read_mask(path: Path) -> tuple[VoxelMask, nib.Nifti1Image]:
    image = nib.load(path)
    return np.asarray(image.dataobj) > 0, image


def write_mask_like(mask: VoxelMask | UInt8[np.ndarray, "x y z"], reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, path)


def require_same_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if reference.shape != other.shape or not np.allclose(reference.affine, other.affine):
        raise ValueError(f"{name} is not on the voxel grid of the first view")


def read_view_masks(job: PostProcessJob) -> tuple[list[VoxelMask], nib.Nifti1Image]:
    "the mask of each view, and the image of the first view, whose grid the others must share"
    first_mask, reference = read_mask(view_mask_path(job.mask_path, job.views[0]))
    masks = [first_mask]
    for view in job.views[1:]:
        path = view_mask_path(job.mask_path, view)
        mask, image = read_mask(path)
        require_same_grid(reference, image, path.name)
        masks.append(mask)
    return masks, reference


def post_process_fissure_masks(job: PostProcessJob) -> Path:
    masks, image = read_view_masks(job)
    sheet = oriented_sheet(
        np.logical_or.reduce(masks),
        **sheet_parameters(job.sheet, job.slice_step),
    )
    removed = RemovedVoxels(sheet.shape)
    trimmed = apply_island_rules(job.island_rules, sheet, image.affine, removed)
    write_mask_like(removed.labels, image, job.removed_path)
    report = PostProcessReport(removed=removed.steps, islands=trimmed.report)
    job.report_path.write_text(report.model_dump_json(indent=2))
    write_mask_like(trimmed.kept, image, job.sheet_path)
    return job.sheet_path
