"""Clean the per-view fissure masks and blend them into one thin sheet.

Why:
- The masks from `make_fissure_mask` come from lines traced slice by slice, so they have false lines and look ragged in 3D.
  - False lines are traces of the body outline, the table, the diaphragm and the hilum.
  - Ragged means a stack of separate lines with holes between them, most of all where a fissure lies almost flat in the slice plane.
- Each view is blind to some fissures, so the views are blended after cleaning and not before.

How:
0. The lung lobes are segmented from the scan (`segmentation.py`), unless the lobes file exists already.
1. The lungs are the union of the lobes, and a region is made of them with a margin of `lung_margin_mm`.
2. The lines of each view are judged on their own (`imaging/lines.py`).
   - A line with less than `min_share_inside` of its voxels in the region is removed, any other line is kept whole.
   - Those are the outline of the body and the table, which a few views trace.
   - Removed lines of all views are saved as one mask, so they can be checked.
3. The kept masks are OR-ed into one mask.
4. The union is blurred along the sheet and thinned to one voxel (`imaging/sheet.py`).
5. Pieces of the sheet under `min_piece_voxels` are dropped and saved as `removed_pieces.nii.gz`.
   - They are stubs and specks left over from lines that only one view traced, and a fissure is a large surface.

Annotating every `slice_step`-th slice cuts the cost of the model by about that factor.
- Fewer slices give less density on a sheet and longer holes between the lines.
- So the sheet settings that the config leaves out follow the step (`sheet_parameters`):
  - The density cutoff is divided by the step.
  - The neighbourhood and the blur along the sheet grow by one voxel per extra step.
- The rule was found on one scan by comparing the sheet with the sheet of the full run.
  - At a step of 3 it matched it, with recall 0.99 and precision 1.0 within 2.4 mm.
  - The settings of the full run lost 11% to 13% of the sheet at that step.
- Settings that the config sets are never changed.

The lobes file is kept, so a second run on the same scan does not segment again.
A segmenter other than TotalSegmentator can be passed in, it only has to follow `Segmenter`.
Lines are judged by the lungs and not by the lobe interfaces, which an earlier version used.
- The interfaces of the segmentation are off in severe emphysema.
  - A real left oblique fissure was 18 mm from its interface, and the old rule removed it.
- On the other scans the interface rule removed the same outlines of the body and little else, since the later filters drop lone short lines anyway.
"""

from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Bool
from pydantic import Field

from fissure_annotation.fissure_mask import view_mask_path
from fissure_annotation.imaging.lines import (
    keep_large_pieces,
    keep_lines_mostly_inside,
)
from fissure_annotation.imaging.lobes import distance_to_mask_mm
from fissure_annotation.imaging.sheet import (
    DEFAULT_MIN_STRENGTH,
    DEFAULT_SIGMA_ACROSS_VOXELS,
    DEFAULT_SIGMA_ALONG_VOXELS,
    DEFAULT_SIGMA_PCA_VOXELS,
    oriented_sheet,
)
from fissure_annotation.imaging.slices import View
from fissure_annotation.segmentation import (
    LUNG_LOBES,
    Segmenter,
    TotalSegmentatorSettings,
    run_totalsegmentator,
)
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


class PostProcessJob(ExperimentSettings):
    nifti: Path  # the scan the masks were made from, the lobes are segmented from it
    mask_path: Path  # the combined mask of make_fissure_mask, the mask of each view is found next to it
    views: list[View] = ["axial"]
    lobes: Path  # where the lung lobes are saved, and read from when the file exists already
    out_dir: Path
    lung_margin_mm: float = 5.0  # the lungs as the segmentation has them are a little tight, and the bases are not exact
    min_share_inside: float = Field(
        default=0.5, ge=0, le=1
    )  # a line with less than this share of its voxels near the lungs is removed
    slice_step: int = Field(
        default=1, ge=1
    )  # the model annotated every slice_step-th slice of each view, 2 for every other slice
    min_piece_voxels: int = Field(
        default=150, ge=0
    )  # pieces of the final sheet under this size are dropped, 0 keeps all of them
    sheet: SheetSettings = SheetSettings()
    totalsegmentator: TotalSegmentatorSettings = TotalSegmentatorSettings()

    @property
    def sheet_path(self) -> Path:
        return self.out_dir / "sheet.nii.gz"

    @property
    def removed_lines_path(self) -> Path:
        return self.out_dir / "removed_lines.nii.gz"

    @property
    def removed_pieces_path(self) -> Path:
        return self.out_dir / "removed_pieces.nii.gz"


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


def write_mask_like(mask: VoxelMask, reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, path)


def require_same_grid(
    reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str
) -> None:
    if reference.shape != other.shape or not np.allclose(
        reference.affine, other.affine
    ):
        raise ValueError(f"{name} is not on the voxel grid of the lobes")


def near_the_lungs(lobes: nib.Nifti1Image, margin_mm: float) -> VoxelMask:
    spacing_mm = tuple(float(s) for s in lobes.header.get_zooms()[:3])
    lungs = np.asarray(lobes.dataobj) > 0
    return distance_to_mask_mm(lungs, spacing_mm) <= margin_mm


def segmented_lobes(job: PostProcessJob, segmenter: Segmenter) -> nib.Nifti1Image:
    if not job.lobes.exists():
        segmenter(job.nifti, LUNG_LOBES, job.lobes)
    return nib.load(job.lobes)


def post_process_fissure_masks(
    job: PostProcessJob, segmenter: Segmenter | None = None
) -> Path:
    segmenter = segmenter or partial(
        run_totalsegmentator, settings=job.totalsegmentator
    )
    lobes = segmented_lobes(job, segmenter)
    near_lungs = near_the_lungs(lobes, job.lung_margin_mm)
    kept_masks, removed_masks = [], []
    for view in job.views:
        path = view_mask_path(job.mask_path, view)
        mask, image = read_mask(path)
        require_same_grid(lobes, image, path.name)
        kept, removed = keep_lines_mostly_inside(mask, near_lungs, job.min_share_inside)
        kept_masks.append(kept)
        removed_masks.append(removed)
    sheet = oriented_sheet(
        np.logical_or.reduce(kept_masks),
        **sheet_parameters(job.sheet, job.slice_step),
    )
    sheet, small_pieces = keep_large_pieces(sheet, job.min_piece_voxels)
    write_mask_like(np.logical_or.reduce(removed_masks), image, job.removed_lines_path)
    write_mask_like(small_pieces, image, job.removed_pieces_path)
    write_mask_like(sheet, image, job.sheet_path)
    return job.sheet_path
