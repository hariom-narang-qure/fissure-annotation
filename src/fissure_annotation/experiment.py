"""Run the waypoint agent over a range of slices of one scan, and write a contact sheet of the results.

- Each slice is an independent conversation, run in parallel, one thread per slice.
- `slice_step` above 1 runs only every slice_step-th slice of the range, so the cost drops by about that factor.
- Everything that distinguishes one run from another (scan, slices, model) is in the YAML config.
- The client comes from `agent.provider`, see `model_client`.
- Output goes under `out_dir`: points/, masks/, overlays/, logs/ and contact_sheet.png (see waypoints.RunFolder).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from jaxtyping import UInt8
from openai import OpenAI
from portkey_ai import Portkey
from pydantic import Field
from tqdm import tqdm

from fissure_annotation.agent import AgentSettings, SliceLog, run_slice
from fissure_annotation.imaging.overlay import contact_sheet, labelled, red_overlay
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.imaging.slices import View, slice_file_name, write_slices
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.waypoints import RunFolder

CONTACT_SHEET_COLUMNS = 3
SIDE_BY_SIDE_COLUMNS = 2


class WaypointExperiment(ExperimentSettings):
    slices_dir: Path
    out_dir: Path
    first_slice: int
    last_slice: int
    slice_step: int = Field(
        default=1, ge=1
    )  # every slice_step-th slice from first_slice, and always last_slice so the end of the range is seen
    workers: int = 16
    agent: AgentSettings = AgentSettings()

    @property
    def slice_indices(self) -> list[int]:
        every_step = range(self.first_slice, self.last_slice + 1, self.slice_step)
        return sorted({*every_step, self.last_slice})


class SliceExport(ExperimentSettings):
    nifti: Path
    out_dir: Path
    view: View = "axial"


def export_slices(config: SliceExport) -> int:
    return write_slices(config.nifti, config.out_dir, config.view)


FLEX_TIMEOUT_SECONDS = 900.0  # flex answers slowly, OpenAI suggests 15 minutes instead of its default 10


def portkey_client() -> Portkey:
    return Portkey(api_key=os.environ["PORTKEY_API_KEY"])


def openai_client() -> OpenAI:
    "max_retries=0, since the SDK would retry a 429 silently and retry.py does it with a log line"
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=FLEX_TIMEOUT_SECONDS)


def model_client(agent: AgentSettings) -> Portkey | OpenAI:
    return openai_client() if agent.provider == "openai" else portkey_client()


def run_slices(config: WaypointExperiment, client) -> list[SliceLog]:
    "slices that already have a log are skipped, so a run that failed halfway is finished by running it again"
    folder = RunFolder(root=config.out_dir)
    pending = [
        z
        for z in config.slice_indices
        if not folder.log_path(config.slices_dir / slice_file_name(z)).exists()
    ]
    with ThreadPoolExecutor(config.workers) as pool:
        logs = pool.map(
            lambda z: run_slice(client, config.agent, folder, config.slices_dir, z),
            pending,
        )
        return list(tqdm(logs, total=len(pending), desc=f"{config.agent.view} slices", unit="slice"))


def run_experiment(config: WaypointExperiment, client) -> list[SliceLog]:
    logs = run_slices(config, client)
    write_sheets(config)
    return logs


def write_sheets(config: WaypointExperiment) -> None:
    "both sheets are built from the files of a finished run, so they can be remade without calling the model"
    folder = RunFolder(root=config.out_dir)
    save_png(sheet_of_results(config, folder), config.out_dir / "contact_sheet.png")
    save_png(side_by_side_sheet(config, folder), config.out_dir / "side_by_side.png")


def crop_to_lungs(
    config: WaypointExperiment, tile: UInt8[np.ndarray, "h w c"]
) -> UInt8[np.ndarray, "h w c"]:
    x0, y0, x1, y1 = config.agent.overlay_box_for(tile.shape[:2])
    return np.ascontiguousarray(tile[y0:y1, x0:x1])


def original_tile(
    config: WaypointExperiment, slice_index: int
) -> UInt8[np.ndarray, "h w 3"]:
    image = read_gray(config.slices_dir / slice_file_name(slice_index))
    return labelled(
        crop_to_lungs(config, np.stack([image] * 3, axis=-1)),
        f"z={slice_index} original",
    )


def result_tile(
    config: WaypointExperiment, folder: RunFolder, slice_index: int
) -> UInt8[np.ndarray, "h w 3"]:
    "the traced slice, or the plain slice when the model traced nothing, cropped to the lungs"
    slice_png = config.slices_dir / slice_file_name(slice_index)
    image = read_gray(slice_png)
    mask_path = folder.mask_path(slice_png)
    tile = (
        red_overlay(image, read_gray(mask_path) > 0)
        if mask_path.exists()
        else np.stack([image] * 3, axis=-1)
    )
    return labelled(crop_to_lungs(config, tile), f"z={slice_index}")


def sheet_of_results(
    config: WaypointExperiment, folder: RunFolder
) -> UInt8[np.ndarray, "h w 3"]:
    tiles = [result_tile(config, folder, z) for z in config.slice_indices]
    return contact_sheet(tiles, CONTACT_SHEET_COLUMNS)


def side_by_side_sheet(
    config: WaypointExperiment, folder: RunFolder
) -> UInt8[np.ndarray, "h w 3"]:
    "each tile is the original slice next to its annotated slice"
    pairs = [
        np.hstack([original_tile(config, z), result_tile(config, folder, z)])
        for z in config.slice_indices
    ]
    return contact_sheet(pairs, SIDE_BY_SIDE_COLUMNS)
