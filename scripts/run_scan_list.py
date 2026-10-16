#!/usr/bin/env python
"""The fissure pipeline over a list of scans. Settings come from a YAML in configs/, flags go on top.
Look at the plan first, it calls nothing and prints the slices and the estimated cost:

    uv run python scripts/run_scan_list.py --config configs/scan_list.yaml --dry-run true

Then try a few scans before the whole list:

    uv run --env-file .env python scripts/run_scan_list.py --config configs/scan_list.yaml --limit 2

Needs in the environment, for the model calls (not for the plan):
- PORTKEY_API_KEY, which `--env-file .env` provides.
"""

import sys

from fissure_annotation.experiment import portkey_client
from fissure_annotation.scan_list import ScanListJob, plan_lines, run_scan_list
from fissure_annotation.settings import load_settings

if __name__ == "__main__":
    job = load_settings(ScanListJob, sys.argv[1:])
    if job.dry_run:
        print("\n".join(plan_lines(job)))
        sys.exit(0)
    statuses = run_scan_list(job, portkey_client())
    failed = [uid for uid, status in statuses.items() if status.state == "failed"]
    print(f"ran {len(statuses)} scans, {len(failed)} failed, see status.json in their folders")
    sys.exit(1 if failed else 0)
