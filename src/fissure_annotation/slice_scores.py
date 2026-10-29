"""Score each annotated slice of a run against a manual lobe segmentation, see imaging/line_scores.py.

How:
- The lobes are cut into the same planes as the slice PNGs (slices.slice_view), so a lobe pixel sits under the slice pixel it belongs to.
- The lobes must be on the grid of the scan the run used, which prepare_flex_step4.py makes sure of for LOCCA.
- A slice the agent ran on but traced nothing on has no mask file, and scores as an empty mask.
- Only slices with a log are scored, so a run that stopped halfway scores the slices it finished.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from pydantic import BaseModel

from fissure_annotation.imaging.line_scores import DEFAULT_TOLERANCE_MM, score_lines
from fissure_annotation.imaging.png import read_gray
from fissure_annotation.imaging.slices import View, slice_view, view_pixel_spacing
from fissure_annotation.waypoints import RunFolder


class SliceScore(BaseModel):
    view: View
    slice_index: int
    traced_px: int
    traced_near_reference_px: int
    reference_px: int
    reference_near_traced_px: int


class ViewTotals(BaseModel):
    "pixel-weighted over the slices, so a slice with a long fissure counts more than one with a stub"

    view: str
    slices: int
    precision: float
    recall: float


def slice_index_of(log_path: Path) -> int:
    return int(log_path.stem.removeprefix("slice_"))


def scored_slices(run: RunFolder) -> list[int]:
    return sorted(slice_index_of(path) for path in (run.root / "logs").glob("slice_*.json"))


def traced_mask(run: RunFolder, slices_dir: Path, slice_index: int, shape: tuple[int, int]) -> np.ndarray:
    path = run.mask_path(slices_dir / f"slice_{slice_index:04d}.png")
    return read_gray(path) > 0 if path.exists() else np.zeros(shape, dtype=bool)


def score_view(
    lobes_path: Path,
    view_dir: Path,
    view: View,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    slice_indices: list[int] | None = None,
) -> list[SliceScore]:
    "view_dir holds slices/ and run/, as fissure_mask.view_masks writes them"
    run = RunFolder(root=view_dir / "run")
    indices = scored_slices(run) if slice_indices is None else slice_indices
    return score_run(lobes_path, view_dir / "slices", run, view, indices, tolerance_mm)


def score_run(
    lobes_path: Path,
    slices_dir: Path,
    run: RunFolder,
    view: View,
    slice_indices: list[int],
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> list[SliceScore]:
    image = nib.as_closest_canonical(nib.load(lobes_path))
    lobes = np.asarray(image.dataobj).astype(np.int16)
    spacing = view_pixel_spacing(tuple(float(s) for s in image.header.get_zooms()[:3]), view)
    scores = []
    for z in slice_indices:
        lobe_slice = slice_view(lobes, view, z)
        traced = traced_mask(run, slices_dir, z, lobe_slice.shape)
        score = score_lines(traced, lobe_slice, spacing, tolerance_mm)
        scores.append(SliceScore(view=view, slice_index=z, **score._asdict()))
    return scores


def ratio(part: int, whole: int) -> float:
    return part / whole if whole else float("nan")


def totals(view: str, scores: list[SliceScore]) -> ViewTotals:
    return ViewTotals(
        view=view,
        slices=len(scores),
        precision=ratio(sum(s.traced_near_reference_px for s in scores), sum(s.traced_px for s in scores)),
        recall=ratio(sum(s.reference_near_traced_px for s in scores), sum(s.reference_px for s in scores)),
    )
