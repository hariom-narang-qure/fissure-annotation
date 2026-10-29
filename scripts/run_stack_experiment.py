#!/usr/bin/env python
"""Run the stack agent on the stacks of a config at every stack size, then score it against the one-slice agent.

    uv run --env-file .env python scripts/run_stack_experiment.py --config configs/stack_sweep.yaml

With `--score-only true` nothing is called, the finished conversations are scored again.
Writes scores.csv into out_dir and prints one summary per stack size.
"""

import sys

from fissure_annotation.experiment import model_client
from fissure_annotation.settings import load_settings
from fissure_annotation.stack_experiment import StackExperiment, all_score_rows, run_stack_experiment, summary_lines, write_score_rows


class StackExperimentJob(StackExperiment):
    score_only: bool = False


if __name__ == "__main__":
    job = load_settings(StackExperimentJob, sys.argv[1:])
    if not job.score_only:
        run_stack_experiment(job, model_client(job.agent))
    rows = all_score_rows(job)
    write_score_rows(rows, job.out_dir / "scores.csv")
    print("\n".join(summary_lines(job, rows)))
