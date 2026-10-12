"""Run the waypoint agent over a range of slices of one scan, and write a contact sheet of the results.

- Each slice is an independent conversation, run in parallel, one thread per slice.
- Everything that distinguishes one run from another (scan, slices, model) is in the YAML config.
- Output goes under `out_dir`: points/, masks/, overlays/, logs/ and contact_sheet.png (see waypoints.RunFolder).
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from jaxtyping import UInt8
from portkey_ai import Portkey

from fissure_annotation.agent import AgentSettings, SliceLog, run_slice
from fissure_annotation.imaging.overlay import contact_sheet, labelled, red_overlay
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.imaging.slices import slice_file_name, write_slices
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.waypoints import RunFolder

CONTACT_SHEET_COLUMNS = 3


class WaypointExperiment(ExperimentSettings):
    slices_dir: Path
    out_dir: Path
    first_slice: int
    last_slice: int
    agent: AgentSettings = AgentSettings()

    @property
    def slice_indices(self) -> list[int]:
        return list(range(self.first_slice, self.last_slice + 1))


class SliceExport(ExperimentSettings):
    nifti: Path
    out_dir: Path


def export_slices(config: SliceExport) -> int:
    return write_slices(config.nifti, config.out_dir)


def portkey_client() -> Portkey:
    return Portkey(api_key=os.environ["PORTKEY_API_KEY"])


def run_experiment(config: WaypointExperiment, client) -> list[SliceLog]:
    folder = RunFolder(root=config.out_dir)
    with ThreadPoolExecutor(len(config.slice_indices)) as pool:
        logs = list(pool.map(lambda z: run_slice(client, config.agent, folder, config.slices_dir, z), config.slice_indices))
    save_png(sheet_of_results(config, folder), config.out_dir / "contact_sheet.png")
    return logs


def result_tile(config: WaypointExperiment, folder: RunFolder, slice_index: int) -> UInt8[np.ndarray, "h w 3"]:
    "the traced slice, or the plain slice when the model traced nothing, cropped to the lungs"
    slice_png = config.slices_dir / slice_file_name(slice_index)
    image = read_gray(slice_png)
    mask_path = folder.mask_path(slice_png)
    tile = red_overlay(image, read_gray(mask_path) > 0) if mask_path.exists() else np.stack([image] * 3, axis=-1)
    x0, y0, x1, y1 = config.agent.overlay_box
    return labelled(np.ascontiguousarray(tile[y0:y1, x0:x1]), f"z={slice_index}")


def sheet_of_results(config: WaypointExperiment, folder: RunFolder) -> UInt8[np.ndarray, "h w 3"]:
    tiles = [result_tile(config, folder, z) for z in config.slice_indices]
    return contact_sheet(tiles, CONTACT_SHEET_COLUMNS)
