#!/usr/bin/env python
"""Scrolling agent over the picked slices. Settings come from a YAML in configs/, flags go on top:

    uv run --env-file .env python scripts/run_scroll.py --config configs/scroll_current.yaml --only scan2
    uv run --env-file .env python scripts/run_scroll.py --config configs/scroll_current.yaml --agent.plain-images true
"""
import json
import sys

from fissure_annotation.scroll_experiment import ScrollExperimentConfig, run_scroll_experiment
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(json.dumps(run_scroll_experiment(load_settings(ScrollExperimentConfig, sys.argv[1:])), indent=1, default=str))
