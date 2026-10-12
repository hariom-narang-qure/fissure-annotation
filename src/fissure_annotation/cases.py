"""Which scans and slices an experiment runs on, shared by every experiment runner.

- `cases` is a JSON file of {name: [scan, lobe_dir]}.
- The slices of a case are its `picks.json` under the experiment output, written by `picker`.
"""
import json
from pathlib import Path

from fissure_annotation.config import EXPERIMENT_OUT as OUT
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.slicing import Case, load_case

Pick = tuple[str, int]
"""(plane, slice index)"""


class CasesSettings(ExperimentSettings):
    config: Path | None = None
    cases: Path = Path("work/cases.json")
    only: str = ""
    """Only the cases whose name contains this."""
    workers: int = 5
    max_jobs: int | None = None
    """Only the first n jobs, for a fast smoke run."""


def case_files(settings: CasesSettings) -> dict[str, tuple[str, str]]:
    cases = json.loads(settings.cases.read_text())
    return {name: (scan, lobe_dir) for name, (scan, lobe_dir) in cases.items() if settings.only in name}


def read_picks(name: str) -> list[Pick]:
    return [(plane, idx) for plane, idx in json.loads((OUT / name / "picks.json").read_text())]


def picked_slices(settings: CasesSettings) -> list[tuple[Case, str, str, int]]:
    "(case, case name, plane, slice index) for every picked slice, limited by `max_jobs`"
    jobs = []
    for name, (scan, lobe_dir) in case_files(settings).items():
        case = load_case(scan, lobe_dir)
        jobs += [(case, name, plane, idx) for plane, idx in read_picks(name)]
    return jobs[: settings.max_jobs]
