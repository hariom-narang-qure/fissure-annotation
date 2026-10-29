#!/usr/bin/env python
"""Score the per-slice lines of a scan run against its lobes (slice_scores.py), and write scores.csv into the run.

    uv run python scripts/score_slices.py --run-dir work/scans/locca_covid_028/flex_step4

The run folder needs lobes.nii.gz and fissures_work/<view>/, which scan_list.py makes.
"""

import csv
import sys
from pathlib import Path

from fissure_annotation.imaging.slices import View
from fissure_annotation.settings import ExperimentSettings, load_settings
from fissure_annotation.slice_scores import score_view, totals


class ScoreRun(ExperimentSettings):
    run_dir: Path
    views: list[View] = ["axial", "coronal", "sagittal"]
    tolerance_mm: float = 3.0


if __name__ == "__main__":
    job = load_settings(ScoreRun, sys.argv[1:])
    scores = {
        view: score_view(job.run_dir / "lobes.nii.gz", job.run_dir / "fissures_work" / view, view, job.tolerance_mm)
        for view in job.views
    }
    rows = [score.model_dump() for view_scores in scores.values() for score in view_scores]
    with (job.run_dir / "scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for view, view_scores in [*scores.items(), ("all", [s for v in scores.values() for s in v])]:
        print(totals(view, view_scores).model_dump_json())
