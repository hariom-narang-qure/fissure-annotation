#!/usr/bin/env python
"""A NIfTI scan to a NIfTI fissure mask in the same space. Settings come from a YAML in configs/, flags go on top.
The agent calls the model once per slice, so try a small range first:

    uv run --env-file .env python scripts/make_fissure_mask.py --config configs/make_fissure_mask.yaml \\
        --nifti scan.nii.gz --mask-path out/scan_fissures.nii.gz --first-slice 165 --last-slice 179
"""

import sys

from fissure_annotation.experiment import portkey_client
from fissure_annotation.fissure_mask import FissureMaskJob, make_fissure_mask
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    print(
        make_fissure_mask(load_settings(FissureMaskJob, sys.argv[1:]), portkey_client())
    )
