#!/usr/bin/env python
"""The per-scan agent over a set of scans. Settings come from a YAML in configs/, flags go on top:

    uv run --env-file .env python scripts/run_scan.py --config configs/scan_agent.yaml --only scan2
"""
import json
import sys

from fissure_annotation.scan_experiment import ScanExperimentConfig, run_scan_experiment
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(json.dumps(run_scan_experiment(load_settings(ScanExperimentConfig, sys.argv[1:])), indent=1, default=str))
