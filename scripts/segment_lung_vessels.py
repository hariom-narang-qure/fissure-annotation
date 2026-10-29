#!/usr/bin/env python
"""Segment the lung vessels of the scans of a config with TotalSegmentator, for the vessel dots of the stack agent.

    uv run python scripts/segment_lung_vessels.py --config configs/segment_lung_vessels.yaml

A scan whose output exists is skipped. Every output is checked to be on its scan's grid.
"""

import sys

from fissure_annotation.settings import load_settings
from fissure_annotation.vessel_hints import LungVesselJob, segment_lung_vessels

if __name__ == "__main__":
    segment_lung_vessels(load_settings(LungVesselJob, sys.argv[1:]))
