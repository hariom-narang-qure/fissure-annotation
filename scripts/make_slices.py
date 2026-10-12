#!/usr/bin/env python
"""A NIfTI scan to an axial PNG series. Settings come from a YAML in configs/:

    uv run python scripts/make_slices.py --config configs/make_slices.yaml
"""
import sys

from fissure_annotation.experiment import SliceExport, export_slices
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(export_slices(load_settings(SliceExport, sys.argv[1:])), "slices written")
