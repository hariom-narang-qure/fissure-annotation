#!/usr/bin/env python
"""Upload the scans of a scan list and their fissure sheets to a Lucid project. Settings come from a YAML in configs/, flags go on top.
Look at what would be sent first, it stages the files and runs `aws s3 sync --dryrun`, and calls neither S3 nor Lucid:

    uv run python scripts/upload_to_lucid.py --config configs/lucid_upload.yaml --dry-run true

Then upload for real:

    uv run python scripts/upload_to_lucid.py --config configs/lucid_upload.yaml

Needs in the environment (direnv in the salvo folder sets these, and puts `salvo-client` on the PATH):
- SALVO_BUCKET_PREFIX: the S3 prefix of the user, there is no default and the script stops without it.
- SALVO_BASE_URL: https://lucid.qure.ai for prod.
- SALVO_TOKEN, the account API token, or SALVO_USERNAME and SALVO_PASSWORD instead.
- AWS credentials for the `aws` CLI, from ~/.aws or AWS_* variables, with write access to the bucket in the config.
Not needed for the dry run: SALVO_BASE_URL, SALVO_TOKEN, SALVO_USERNAME and SALVO_PASSWORD.
"""

import sys

from fissure_annotation.lucid_upload import LucidUploadJob, upload_to_lucid
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    left_out = upload_to_lucid(load_settings(LucidUploadJob, sys.argv[1:]))
    print(f"{len(left_out)} series left out, a finding is missing: {left_out}")
