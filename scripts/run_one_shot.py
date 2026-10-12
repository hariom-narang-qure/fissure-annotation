#!/usr/bin/env python
"""The one-shot prompt variants over the picked slices. Settings come from a YAML in configs/, flags go on top:

    uv run --env-file .env python scripts/run_one_shot.py --config configs/one_shot_e.yaml --only scan2
"""
import json
import sys

from fissure_annotation.experiment import OneShotConfig, run_one_shot_experiment
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(json.dumps(run_one_shot_experiment(load_settings(OneShotConfig, sys.argv[1:])), indent=1, default=str))
