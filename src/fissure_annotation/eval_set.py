"""The eval set: one folder per scan, holding the scan and its ground truth.

Layout of a case folder:
- `scan.nii.gz`: the CT. The harness renders every slice it needs from it.
- `gt/<view>/slice_<NNNN>_mask.png`: the reference lines of one slice, white on black, on the grid of the rendered slice.
  - An all-black mask says the slice has no visible fissure. That is ground truth too.
- `meta.json`, read as `EvalMeta`:
  - `tests`: the tests the case is for, such as `["slice"]`.
  - `slices`: every slice that has ground truth, as a view and an index.
  - `source`: where the ground truth came from, in words.

How the harness uses it:
- For each kind of test, it gathers the cases whose meta lists it (`cases_for`), then runs that test on them.
- How a test schedules the agent around the ground truth slices is the test's business, not the data's.
- The folders are tracked with DVC (data/eval.dvc). `uv run dvc pull` fetches them.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from fissure_annotation.imaging.slices import View, slice_file_name

TestKind = Literal["slice"]
META_FILE = "meta.json"


class GroundTruthSlice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    view: View
    index: int


class EvalMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tests: list[TestKind]
    slices: list[GroundTruthSlice]
    source: str = ""


def gt_file_name(index: int) -> str:
    return slice_file_name(index).replace(".png", "_mask.png")


class EvalCase:
    def __init__(self, folder: Path):
        self.folder = folder
        self.meta = EvalMeta.model_validate_json((folder / META_FILE).read_text())

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def scan_path(self) -> Path:
        return self.folder / "scan.nii.gz"

    def gt_path(self, view: View, index: int) -> Path:
        return self.folder / "gt" / view / gt_file_name(index)

    def views(self) -> list[View]:
        return sorted({gt.view for gt in self.meta.slices})

    def indices_of(self, view: View) -> list[int]:
        return sorted(gt.index for gt in self.meta.slices if gt.view == view)


def cases_for(eval_root: Path, test: TestKind) -> list[EvalCase]:
    "every case folder under eval_root whose meta lists the test, in name order"
    cases = [EvalCase(meta.parent) for meta in sorted(eval_root.glob(f"*/{META_FILE}"))]
    return [case for case in cases if test in case.meta.tests]
