"""The agents that annotate the chosen slices of one view, picked in the config by `kind`.

Why:
- The pipeline (fissure_mask.py) only needs the masks of the slices, not how they were made.
- So each agent variant is an annotator, and a config chooses one without flags in the pipeline.

How:
- `OneSliceAnnotator` runs one conversation per slice (experiment.run_slices).
- `StackAnnotator` cuts the slices into stacks of `size` neighbours and runs one conversation per stack (stack_agent.py).
  1. The stacks follow each other without overlap, so every slice is annotated once.
     - The last stack of a view can be smaller.
     - The pipeline always annotates the last slice, so the last stack can have an uneven gap. It is outside the lungs.
  2. The guides come from automatic lobes on the scan's grid (fissure_hints.py), never from manual lobes.
  3. A stack with a log is not run again, so a stopped run is finished by running it again.
- Both write the files of waypoints.RunFolder, so the masks are read the same way.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import nibabel as nib
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from fissure_annotation.agent import AgentSettings
from fissure_annotation.experiment import WaypointExperiment, run_slices
from fissure_annotation.fissure_hints import OUTLINE_COLOUR_OF_FISSURE, write_hint_slices
from fissure_annotation.imaging.png import read_gray
from fissure_annotation.imaging.slices import VIEW_AXIS, View, slice_file_name
from fissure_annotation.stack_agent import DEFAULT_PROMPT, OverviewGuides, StackFolder, run_stack


class ViewWork(BaseModel):
    "what an annotator needs for one view of one scan"

    scan: Path
    lobes: Path | None  # automatic lobes on the scan's grid, for annotators that draw guides
    view: View
    view_dir: Path  # holds slices/, and the annotator writes into run/
    first_slice: int
    last_slice: int
    slice_step: int
    workers: int
    agent: AgentSettings

    @property
    def one_slice_run(self) -> WaypointExperiment:
        return WaypointExperiment(
            slices_dir=self.view_dir / "slices",
            out_dir=self.view_dir / "run",
            first_slice=self.first_slice,
            last_slice=self.last_slice,
            slice_step=self.slice_step,
            workers=self.workers,
            agent=self.agent,
        )

    @property
    def slices_dir(self) -> Path:
        return self.one_slice_run.slices_dir

    @property
    def run_dir(self) -> Path:
        return self.one_slice_run.out_dir

    @property
    def slice_indices(self) -> list[int]:
        "every slice_step-th slice of the range, and always the last one"
        return self.one_slice_run.slice_indices


class OneSliceAnnotator(BaseModel):
    kind: Literal["one_slice"] = "one_slice"

    def annotate(self, work: ViewWork, client) -> None:
        run_slices(work.one_slice_run, client)


def slice_step_mm(scan: Path, view: View, slice_step: int) -> float:
    "the distance between neighbouring slices of a stack, along the axis the view steps through"
    spacing = nib.as_closest_canonical(nib.load(scan)).header.get_zooms()
    return float(spacing[VIEW_AXIS[view]]) * slice_step


def overview_guides(bands_dir: Path, slice_indices: list[int]) -> OverviewGuides:
    bands = {z: read_gray(bands_dir / slice_file_name(z)) for z in slice_indices}
    return OverviewGuides(bands, OUTLINE_COLOUR_OF_FISSURE)


def stacks_of(slice_indices: list[int], size: int) -> list[list[int]]:
    return [slice_indices[start : start + size] for start in range(0, len(slice_indices), size)]


class StackAnnotator(BaseModel):
    kind: Literal["stack"] = "stack"
    size: int = Field(default=5, ge=1)
    prompt: Path = DEFAULT_PROMPT  # it must explain the guides, see prompts/stack_agent.txt
    hint_band_mm: float = 5.0  # each side of a lobe boundary, so a band is about twice as wide

    @field_validator("prompt")
    @classmethod
    def prompt_exists(cls, prompt: Path) -> Path:
        if not prompt.exists():
            raise ValueError(f"no prompt at {prompt}")
        return prompt

    def bands_root(self, work: ViewWork) -> Path:
        "next to the view folders, with a subfolder per view, as write_hint_slices lays them out"
        return work.view_dir.parent / "fissure_bands"

    def write_guides(self, work: ViewWork) -> None:
        if work.lobes is None:
            raise ValueError("the stack annotator draws its guides from lobes, and the job has none")
        write_hint_slices(work.lobes, work.scan, self.bands_root(work), [work.view], self.hint_band_mm)

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
            overview_guides(self.bands_root(work) / work.view, stack),
        )

    def annotate(self, work: ViewWork, client) -> None:
        self.write_guides(work)
        pending = [stack for stack in stacks_of(work.slice_indices, self.size) if not self.is_done(work, stack)]
        with ThreadPoolExecutor(work.workers) as pool:
            results = pool.map(lambda stack: self.run_one_stack(work, stack, client), pending)
            list(tqdm(results, total=len(pending), desc=f"{work.view} stacks", unit="stack"))


Annotator = OneSliceAnnotator | StackAnnotator
