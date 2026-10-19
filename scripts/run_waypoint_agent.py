#!/usr/bin/env python
"""The waypoint agent over a range of slices. Settings come from a YAML in configs/, flags go on top:

uv run --env-file .env python scripts/run_waypoint_agent.py --config configs/waypoint_agent.yaml --agent.max-steps 20
"""

import sys

from fissure_annotation.experiment import (
    WaypointExperiment,
    model_client,
    run_experiment,
)
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    config = load_settings(WaypointExperiment, sys.argv[1:])
    for log in run_experiment(config, model_client(config.agent)):
        print(log.model_dump_json(exclude={"calls"}))
