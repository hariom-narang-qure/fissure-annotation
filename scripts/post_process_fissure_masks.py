#!/usr/bin/env python
"""Clean the per-view fissure masks of make_fissure_mask and blend them into one thin sheet.
Settings come from a YAML in configs/, flags go on top.
The lung lobes are segmented from the scan with TotalSegmentator when the --lobes file does not exist, and reused when it does:

    uv run python scripts/post_process_fissure_masks.py --config configs/post_process_fissure_masks.yaml \\
        --nifti scan.nii.gz --mask-path out/scan_fissures.nii.gz --lobes out/lobes.nii.gz --out-dir out/post
"""

import sys

from fissure_annotation.postprocess import PostProcessJob, post_process_fissure_masks
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(post_process_fissure_masks(load_settings(PostProcessJob, sys.argv[1:])))
