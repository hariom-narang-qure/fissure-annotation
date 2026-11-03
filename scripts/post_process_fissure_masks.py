#!/usr/bin/env python
"""Blend the per-view fissure masks of make_fissure_mask into one thin sheet, and trim it with the island rules.
Settings come from a YAML in configs/, flags go on top:

    uv run python scripts/post_process_fissure_masks.py --config configs/post_process_fissure_masks.yaml \\
        --mask-path out/scan_fissures.nii.gz --out-dir out/post
"""

import sys

from fissure_annotation.postprocess import PostProcessJob, post_process_fissure_masks
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(post_process_fissure_masks(load_settings(PostProcessJob, sys.argv[1:])))
