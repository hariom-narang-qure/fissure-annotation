"""Every voxel the post-process removed from one scan, as one label map with a name per label.

Why:
- A person checking a scan in the viewer wants every removed voxel, and the step that removed it.
- One label map with a legend loads as one overlay, with a name and a text per label.

How:
- Each step adds what it removed as it runs, and takes the next label, starting at 1.
  - So the labels follow the order the steps ran, with no list of steps kept anywhere else.
  - 0 is background in a NIfTI label map, so it means "not removed".
  - A step gets a label even when it removed nothing, so the legend lists every step that ran.
- A voxel removed by two steps keeps the label of the first step.
  - Each rule only sees what the rules before it kept, so today this never happens. It keeps the map well defined for a step that does.
"""

import numpy as np
from jaxtyping import Bool, UInt8
from pydantic import BaseModel


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
