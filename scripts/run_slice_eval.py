#!/usr/bin/env python
"""The slice test of the eval set (slice_eval.py). Settings come from a YAML in configs/, flags go on top.
Fetch the eval set first, then look at the plan, which calls nothing:

    uv run dvc pull
    uv run python scripts/run_slice_eval.py --config configs/slice_eval.yaml --dry-run true

Then try a few stacks before the whole set:

    uv run --env-file .env python scripts/run_slice_eval.py --config configs/slice_eval.yaml --max-stacks 2 --repeats 1

The scores go to scores.csv in out_dir, and the summary is printed.
"""

import sys

from fissure_annotation.experiment import model_client
from fissure_annotation.settings import load_settings
from fissure_annotation.slice_eval import SliceEval, SliceEvalRun, plan_lines, run_slice_eval, score_rows, summary_lines, write_score_rows

if __name__ == "__main__":
    settings = load_settings(SliceEval, sys.argv[1:])
    if settings.dry_run:
        print("\n".join(plan_lines(SliceEvalRun(settings))))
        sys.exit(0)
    run = run_slice_eval(settings, model_client(settings.agent))
    rows = score_rows(run)
    write_score_rows(rows, settings.out_dir / "scores.csv")
    print("\n".join(summary_lines(run, rows)))
