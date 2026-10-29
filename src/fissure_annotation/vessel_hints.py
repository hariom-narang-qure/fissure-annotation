"""Dots on the vessels of each slice, so the agent knows a bright line through one is a vessel, from an automatic vessel segmentation.

Why:
- On the ridge map, a vessel lying in the slice looks like a faint fissure.
  - On weaker models about a third of the traced pixels sat on vessels.
- A blue dot on every vessel cross-section marks those spots, and leaves the pixels around it visible.

How:
1. TotalSegmentator's `lung_vessels` task segments the arteries and the veins on the scan's grid (`segment_lung_vessels`).
   - The grid is checked, since a dot must sit on the pixel of the CT PNG it marks.
2. The vessel mask is cut into slices like the CT PNGs (slices.slice_view), so it has the same orientation.
3. Each slice gets one dot per vessel component, at the pixel farthest inside it (imaging/vessels.py).
   - With `spacing_px` set, long vessels and vessel trees also get dots along their centre lines, at most that far apart.
   - The dots of each slice are written once per scan and view as `slice_XXXX.json`, with a done marker per view.
   - The settings the dots were made with are written next to them, and a folder made with other settings is refused.
4. The agent sees the dots on the slices of its first message only (`stack_agent.VesselDots`).
   - Zooms and trace overlays stay clean, as for the lobe guides.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Bool
from pydantic import BaseModel, Field

from fissure_annotation.fissure_hints import DONE_MARKER, require_same_grid
from fissure_annotation.imaging.slices import VIEW_AXIS, View, slice_file_name, slice_view
from fissure_annotation.imaging.vessels import Colour, Point, vessel_cores, vessel_dots_along_skeleton
from fissure_annotation.segmentation import VESSEL_LABEL_IDS, TotalSegmentatorSettings, run_lung_vessels
from fissure_annotation.settings import ExperimentSettings

VESSEL_BLUE: Colour = (40, 90, 255)  # red is the agent's trace, cyan, yellow and green are the lobe guides


CORE_SETTINGS_FILE = "core_settings.json"


class CoreSettings(BaseModel):
    "what decides where the dots go, so a folder of dots can be checked against it"

    min_area_px: int = Field(default=4, ge=1)  # smaller components are mostly segmentation noise
    spacing_px: float | None = Field(default=None, gt=0)  # None is one dot per component; else dots along each centre line too

    def dots_of(self, mask: Bool[np.ndarray, "h w"]) -> list[Point]:
        if self.spacing_px is None:
            return vessel_cores(mask, self.min_area_px)
        return vessel_dots_along_skeleton(mask, self.min_area_px, self.spacing_px)


class VesselDotSettings(CoreSettings):
    radius_px: float = Field(default=2.0, gt=0)
    colour: Colour = VESSEL_BLUE

    @property
    def core_settings(self) -> CoreSettings:
        return CoreSettings(min_area_px=self.min_area_px, spacing_px=self.spacing_px)


class SliceVesselCores(BaseModel):
    slice_index: int
    cores: list[Point]


def vessel_mask_ras(path: Path) -> Bool[np.ndarray, "x y z"]:
    labels = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj)
    return np.isin(labels, VESSEL_LABEL_IDS)


def cores_path(view_dir: Path, slice_index: int) -> Path:
    return view_dir / slice_file_name(slice_index).replace(".png", ".json")


def write_view_cores(mask: Bool[np.ndarray, "x y z"], view_dir: Path, view: View, settings: CoreSettings) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    for index in range(mask.shape[VIEW_AXIS[view]]):
        cores = settings.dots_of(np.ascontiguousarray(slice_view(mask, view, index)))
        cores_path(view_dir, index).write_text(SliceVesselCores(slice_index=index, cores=cores).model_dump_json())
    (view_dir / CORE_SETTINGS_FILE).write_text(settings.model_dump_json())


def is_written(view_dir: Path, settings: CoreSettings) -> bool:
    if not (view_dir / DONE_MARKER).exists():
        return False
    written = CoreSettings.model_validate_json((view_dir / CORE_SETTINGS_FILE).read_text())
    if written != settings:
        raise ValueError(f"{view_dir} holds dots made with {written}, not {settings}; use another out_dir")
    return True


def write_vessel_cores(vessels_path: Path, scan_path: Path, out_dir: Path, views: list[View], settings: CoreSettings) -> None:
    "a folder per view; a finished folder has a marker file and is not written again"
    pending = [view for view in views if not is_written(out_dir / view, settings)]
    if not pending:
        return
    require_same_grid(vessels_path, scan_path)
    mask = vessel_mask_ras(vessels_path)
    for view in pending:
        write_view_cores(mask, out_dir / view, view, settings)
        (out_dir / view / DONE_MARKER).touch()


def read_vessel_cores(view_dir: Path, slice_indices: list[int]) -> dict[int, list[Point]]:
    return {z: SliceVesselCores.model_validate_json(cores_path(view_dir, z).read_text()).cores for z in slice_indices}


class ScanToSegment(BaseModel):
    scan: Path
    vessels: Path  # the multilabel lung_vessels file to write


class LungVesselJob(ExperimentSettings):
    scans: list[ScanToSegment]
    totalsegmentator: TotalSegmentatorSettings = TotalSegmentatorSettings()


def segment_lung_vessels(job: LungVesselJob) -> None:
    "a scan whose file exists is not segmented again"
    for item in job.scans:
        if not item.vessels.exists():
            run_lung_vessels(item.scan, item.vessels, job.totalsegmentator)
        require_same_grid(item.vessels, item.scan)
