"""What the post-process removed from one scan and why, as one label map and one report.

Why:
- A person checking a scan in the viewer wants every removed voxel, and the step that removed it.
- One label map with a legend loads as one overlay, with a name and a text per label.

How:
- Each step that can remove voxels gets the next label, in the order it ran, starting at 1.
  - 0 is background in a NIfTI label map, so it means "not removed".
  - A step gets a label even when it removed nothing, so the legend lists every step that ran.
- A voxel removed by two steps keeps the label of the first step.
  - Lines are removed from the per-view masks before the sheet is made, so a later step can see the same voxel again.
- The report and the label map are saved in the scan's own output folder, next to the sheet.
"""

import numpy as np
from jaxtyping import Bool, UInt8
from pydantic import BaseModel

from fissure_annotation.island_rules import IslandReport


class RemovalStep(BaseModel):
    label: int  # the value of this step's voxels in removed.nii.gz
    step: str
    voxels: int


class RemovedVoxels:
    def __init__(self, shape: tuple[int, int, int]):
        self.labels: UInt8[np.ndarray, "x y z"] = np.zeros(shape, dtype=np.uint8)
        self.steps: list[RemovalStep] = []

    def add(self, step: str, removed: Bool[np.ndarray, "x y z"]) -> None:
        label = len(self.steps) + 1  # 0 is "not removed"
        first_removed_here = removed & (self.labels == 0)
        self.labels[first_removed_here] = label
        self.steps.append(RemovalStep(label=label, step=step, voxels=int(first_removed_here.sum())))


class PostProcessReport(BaseModel):
    "everything the post-process did to one scan, saved as report.json in the scan's folder"

    removed: list[RemovalStep]  # the legend of removed.nii.gz
    islands: IslandReport

    @property
    def needs_review(self) -> bool:
        return self.islands.needs_review

    def label_of(self, step: str) -> int:
        return next(removal.label for removal in self.removed if removal.step == step)
