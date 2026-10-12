#!/usr/bin/env python
"""The single-slice agent over the picked slices. Settings come from a YAML in configs/, flags go on top:

    uv run --env-file .env python scripts/run_slice_agent.py --config configs/slice_agent.yaml --only scan2
"""
import json
import sys

from fissure_annotation.slice_experiment import SliceExperimentConfig, run_slice_experiment
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(json.dumps(run_slice_experiment(load_settings(SliceExperimentConfig, sys.argv[1:])), indent=1, default=str))
