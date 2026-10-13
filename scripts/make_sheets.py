#!/usr/bin/env python
"""Remake contact_sheet.png and side_by_side.png from the files of a finished run, without calling the model:

uv run python scripts/make_sheets.py --config configs/waypoint_agent.yaml --out-dir work/waypoint_runs/my_run
"""

import sys

from fissure_annotation.experiment import WaypointExperiment, write_sheets
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    write_sheets(load_settings(WaypointExperiment, sys.argv[1:]))
