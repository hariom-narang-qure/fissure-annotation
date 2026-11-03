#!/usr/bin/env python
"""Build data/eval from the gpt-6.1-sol stack runs on two LOCCA scans. It was run once, on 2026-11-03.

What was done:
- The cases are locca_covid_013 and lung_002, the two LOCCA scans of the earlier 40-slice eval.
- The scan is the one those runs used, `work/scans/<case>/flex_step4/scan.nii.gz`.
- The ground truth is sol's traces, repeat 0, as the stack agent saved them (`masks/slice_<NNNN>_mask.png`).
  - 40 slices come from `eval40_step3`: sol on the CT with lobe guides, stacks of 5 slices 3 apart.
  - 20 slices come from `terra_exp90_heldout_sol`: sol on the ridge map without guides, on 4 stacks far from those.
  - A slice sol left empty gets an all-black mask, since "no visible fissure" is ground truth too.
- The traces are reasonable but not reviewed. They are to be cleaned by hand before the next agent experiments.
- The configs of both runs are on the tag experiments/2026-10 (see docs/stack_agent_experiments.md).

    uv run python scripts/build_slice_eval_set.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

from fissure_annotation.eval_set import EvalCase, EvalMeta, GroundTruthSlice
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.imaging.slices import slice_file_name

WORK = Path("work")
OUT = Path("data/eval")
STEP, SIZE = 3, 5
SOURCES = {
    "eval40_step3": "sol on the CT with lobe guides (eval40_step3, repeat 0)",
    "terra_exp90_heldout_sol": "sol on the ridge map without guides (terra_exp90_heldout_sol, repeat 0)",
}
STACK_CENTRES = {  # (run, view, centre) per case, from the two configs on the tag
    "locca_covid_013": [
        ("eval40_step3", "axial", 140), ("eval40_step3", "axial", 272),
        ("eval40_step3", "coronal", 184), ("eval40_step3", "coronal", 288),
        ("eval40_step3", "sagittal", 164), ("eval40_step3", "sagittal", 380),
        ("terra_exp90_heldout_sol", "axial", 205), ("terra_exp90_heldout_sol", "coronal", 236),
    ],
    "lung_002": [
        ("eval40_step3", "axial", 116), ("eval40_step3", "coronal", 212),
        ("terra_exp90_heldout_sol", "axial", 160), ("terra_exp90_heldout_sol", "sagittal", 340),
    ],
}


def stack_slices(centre: int) -> list[int]:
    half = SIZE // 2
    return [centre + STEP * offset for offset in range(-half, half + 1)]


def sol_stack_folder(run: str, case: str, view: str, centre: int) -> Path:
    folder = WORK / "stack_agent" / run / f"{case}_{view}_{centre}" / "size5" / "repeat0"
    if not any((folder / "stack_logs").glob("*.json")):
        raise FileNotFoundError(f"{folder} has no finished conversation")
    return folder


def copy_ground_truth(case: str, run: str, view: str, centre: int, target: EvalCase) -> list[GroundTruthSlice]:
    folder = sol_stack_folder(run, case, view, centre)
    slices_dir = WORK / "scans" / case / "flex_step4" / "fissures_work" / view / "slices"
    for index in stack_slices(centre):
        sol_mask = folder / "masks" / slice_file_name(index).replace(".png", "_mask.png")
        shape = read_gray(slices_dir / slice_file_name(index)).shape
        mask = read_gray(sol_mask) if sol_mask.exists() else np.zeros(shape, dtype=np.uint8)
        save_png(((mask > 0) * 255).astype(np.uint8), target.gt_path(view, index))
    return [GroundTruthSlice(view=view, index=index) for index in stack_slices(centre)]


def build_case(case: str) -> None:
    folder = OUT / case
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile((WORK / "scans" / case / "flex_step4" / "scan.nii.gz").resolve(), folder / "scan.nii.gz")
    (folder / "meta.json").write_text(json.dumps({"tests": [], "slices": []}))  # so EvalCase can be made for its paths
    target = EvalCase(folder)
    slices = [gt for run, view, centre in STACK_CENTRES[case] for gt in copy_ground_truth(case, run, view, centre, target)]
    used = sorted({run for run, _, _ in STACK_CENTRES[case]})
    source = "gpt-6.1-sol traces, not reviewed yet: " + "; ".join(SOURCES[run] for run in used)
    meta = EvalMeta(tests=["slice"], slices=slices, source=source)
    (folder / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n")
    print(f"{case}: {len(slices)} slices")


if __name__ == "__main__":
    for case in STACK_CENTRES:
        build_case(case)
