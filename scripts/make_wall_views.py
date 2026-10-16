#!/usr/bin/env python
"""A scan, a fissure mask and the lobe-interface prior to one PNG per fissure, showing the mask on the fissure as a wall. Settings come from a YAML in configs/:

uv run python scripts/make_wall_views.py --config configs/wall_views.yaml
"""

import sys

from fissure_annotation.fissure_wall_view import WallViewJob, make_wall_views
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    for path in make_wall_views(load_settings(WallViewJob, sys.argv[1:])):
        print(path)
