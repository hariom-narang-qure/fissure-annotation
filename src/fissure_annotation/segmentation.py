"""Label structures of a CT scan, behind one call so the segmenter can be swapped.

Why:
- Post-processing needs the lung lobes, and later steps may need other structures.
- The rest of the package asks for structures by name through `Segmenter` and never knows how they are made.
  - Another segmenter, or another version of TotalSegmentator, only has to follow the same call.

The call:
- `segmenter(scan, structures, out_path)` writes one multilabel NIfTI on the voxel grid of the scan and returns its path.
- The label id of a structure in that file is `LABEL_IDS[structure]`, and 0 is background.
- A structure the segmenter did not find is missing from the file, which is not an error.

How the TotalSegmentator implementation works:
- It runs the `TotalSegmentator` command line with `--ml`, so one multilabel file comes out.
  - `--roi_subset` limits the work to the structures asked for, which is faster.
- It is a command and not an import, because TotalSegmentator brings torch and nnU-Net.
  - They are not dependencies of this repo, and TotalSegmentator is usually installed in an environment of its own.
  - `TotalSegmentatorSettings.command` says how to start it.
- The ids in the file are checked, so a different TotalSegmentator version that numbers the structures differently fails loudly.

Lung vessels (`run_lung_vessels`):
- They come from the `lung_vessels` task, not from `total`, so they have a call of their own.
- From TotalSegmentator 2.x on, that task is the airways, arteries and veins model (task 117).
  - The older vessels and airways model is `lung_vessels_LEGACY` (task 258).
  - Labels of the 117 file: `LUNG_VESSELS_TASK_LABEL_IDS`. The vessels are the arteries and the veins (`VESSEL_LABEL_IDS`).
- The task crops to the lungs and resamples, then writes back on the grid of the scan.
"""

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import nibabel as nib
import numpy as np

from fissure_annotation.imaging.lobes import (
    LEFT_LOWER,
    LEFT_UPPER,
    RIGHT_LOWER,
    RIGHT_MIDDLE,
    RIGHT_UPPER,
)
from fissure_annotation.settings import ExperimentSettings

Structure = Literal[
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]
LUNG_LOBES: list[Structure] = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]
LABEL_IDS: dict[Structure, int] = {
    "lung_upper_lobe_left": LEFT_UPPER,
    "lung_lower_lobe_left": LEFT_LOWER,
    "lung_upper_lobe_right": RIGHT_UPPER,
    "lung_middle_lobe_right": RIGHT_MIDDLE,
    "lung_lower_lobe_right": RIGHT_LOWER,
}
LUNG_VESSELS_TASK_LABEL_IDS = {"lung_airways": 1, "lung_airways_wall": 2, "lung_arteries": 3, "lung_veins": 4}
VESSEL_LABEL_IDS = (LUNG_VESSELS_TASK_LABEL_IDS["lung_arteries"], LUNG_VESSELS_TASK_LABEL_IDS["lung_veins"])
ERROR_TAIL_CHARACTERS = 2000


class Segmenter(Protocol):
    def __call__(
        self, scan: Path, structures: Sequence[Structure], out_path: Path
    ) -> Path: ...


class TotalSegmentatorSettings(ExperimentSettings):
    command: list[str] = [
        "TotalSegmentator"
    ]  # how to start it, for example ["/path/to/venv/bin/python", "-m", "totalsegmentator.bin.TotalSegmentator"]
    device: Literal["gpu", "cpu"] = "gpu"
    gpu_index: int | None = (
        None  # sets CUDA_VISIBLE_DEVICES for the call, the default leaves the environment alone
    )


def totalsegmentator_arguments(
    scan: Path,
    structures: Sequence[Structure],
    out_path: Path,
    settings: TotalSegmentatorSettings,
) -> list[str]:
    return [
        *settings.command,
        "-i",
        str(scan),
        "-o",
        str(out_path),
        "--ml",
        "--task",
        "total",
        "--roi_subset",
        *structures,
        "-d",
        settings.device,
    ]


def environment_for(settings: TotalSegmentatorSettings) -> dict[str, str]:
    if settings.gpu_index is None:
        return dict(os.environ)
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(settings.gpu_index)}


def lung_vessels_arguments(scan: Path, out_path: Path, settings: TotalSegmentatorSettings) -> list[str]:
    return [*settings.command, "-i", str(scan), "-o", str(out_path), "--ml", "--task", "lung_vessels", "-d", settings.device]


def require_only_requested_labels(
    out_path: Path, structures: Sequence[Structure]
) -> None:
    require_only_label_ids(out_path, {LABEL_IDS[structure] for structure in structures})


def require_only_label_ids(out_path: Path, expected: set[int]) -> None:
    found = set(np.unique(np.asarray(nib.load(out_path).dataobj)).tolist()) - {0}
    if not found <= expected:
        raise ValueError(
            f"{out_path.name} has label ids {sorted(found - expected)} that were not asked for, "
            "the segmenter may number the structures differently"
        )


def run_totalsegmentator(
    scan: Path,
    structures: Sequence[Structure],
    out_path: Path,
    settings: TotalSegmentatorSettings = TotalSegmentatorSettings(),
) -> Path:
    run_command(totalsegmentator_arguments(scan, structures, out_path, settings), out_path, settings)
    require_only_requested_labels(out_path, structures)
    return out_path


def run_lung_vessels(scan: Path, out_path: Path, settings: TotalSegmentatorSettings = TotalSegmentatorSettings()) -> Path:
    "the multilabel file of the lung_vessels task; the vessel mask is its VESSEL_LABEL_IDS"
    run_command(lung_vessels_arguments(scan, out_path, settings), out_path, settings)
    require_only_label_ids(out_path, set(LUNG_VESSELS_TASK_LABEL_IDS.values()))
    return out_path


def run_command(arguments: list[str], out_path: Path, settings: TotalSegmentatorSettings) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        arguments,
        env=environment_for(settings),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator failed with exit code {result.returncode}:\n"
            f"{(result.stdout + result.stderr)[-ERROR_TAIL_CHARACTERS:]}"
        )
    if not out_path.exists():
        raise FileNotFoundError(f"TotalSegmentator wrote no file at {out_path}")
