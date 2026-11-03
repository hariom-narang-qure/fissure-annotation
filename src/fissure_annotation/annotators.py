"""The agent that annotates the chosen slices of one view, in stacks of nearby slices.

How:
- `StackAnnotator` cuts the slices into stacks of `size` neighbours and runs one conversation per stack (stack_agent.py).
  1. The stacks follow each other without overlap, so every slice is annotated once.
     - The last stack of a view can be smaller.
     - The pipeline always annotates the last slice, so the last stack can have an uneven gap. It is outside the lungs.
  2. The agent sees the ridge map of each slice, without lobe guides (docs/stack_agent_experiments.md, section 3).
  3. A stack with a log is not run again, so a stopped run is finished by running it again.
- It writes the files of waypoints.RunFolder, so the masks are read as the one-slice agent's were.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
from pydantic import BaseModel, Field
from tqdm import tqdm

from fissure_annotation.agent import AgentSettings
from fissure_annotation.imaging.slices import VIEW_AXIS, View
from fissure_annotation.stack_agent import RIDGE_PROMPT, RidgePictures, StackFolder, run_stack


class ViewWork(BaseModel):
    "what an annotator needs for one view of one scan"

    scan: Path
    view: View
    view_dir: Path  # holds slices/, and the annotator writes into run/
    first_slice: int
    last_slice: int
    slice_step: int
    workers: int
    agent: AgentSettings

    @property
    def slices_dir(self) -> Path:
        return self.view_dir / "slices"

    @property
    def run_dir(self) -> Path:
        return self.view_dir / "run"

    @property
    def slice_indices(self) -> list[int]:
        "every slice_step-th slice of the range, and always the last one"
        every_step = range(self.first_slice, self.last_slice + 1, self.slice_step)
        return sorted({*every_step, self.last_slice})


def slice_step_mm(scan: Path, view: View, slice_step: int) -> float:
    "the distance between neighbouring slices of a stack, along the axis the view steps through"
    spacing = nib.as_closest_canonical(nib.load(scan)).header.get_zooms()
    return float(spacing[VIEW_AXIS[view]]) * slice_step


def stacks_of(slice_indices: list[int], size: int) -> list[list[int]]:
    return [slice_indices[start : start + size] for start in range(0, len(slice_indices), size)]


class StackAnnotator(BaseModel):
    size: int = Field(default=5, ge=1)
    prompt: Path = RIDGE_PROMPT
    # white from 20 lung-window grey levels, the cap of terra experiment 09
    pictures: RidgePictures = RidgePictures(display_max=20)

    def is_done(self, work: ViewWork, stack: list[int]) -> bool:
        return StackFolder(root=work.run_dir).stack_log_path(stack).exists()

    def run_one_stack(self, work: ViewWork, stack: list[int], client) -> None:
        run_stack(
            client,
            work.agent,
            StackFolder(root=work.run_dir),
            work.slices_dir,
            stack,
            slice_step_mm(work.scan, work.view, work.slice_step),
            self.prompt,
            pictures=self.pictures,
        )

    def annotate(self, work: ViewWork, client) -> None:
        pending = [stack for stack in stacks_of(work.slice_indices, self.size) if not self.is_done(work, stack)]
        with ThreadPoolExecutor(work.workers) as pool:
            results = pool.map(lambda stack: self.run_one_stack(work, stack, client), pending)
            list(tqdm(results, total=len(pending), desc=f"{work.view} stacks", unit="stack"))

