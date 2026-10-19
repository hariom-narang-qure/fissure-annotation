#!/usr/bin/env python
"""Run the fissure pipeline on the nine scans added for "Fissure Segmentation Review (v1)", one after the other.

The review project has twenty scans. Eleven ran already, these nine fill in the gaps:
- Emphysema at every severity, and nodules of every size.
- Things that can look like a fissure: bulla walls, thickened septa, airway walls, metal streaks.
- A control whose lungs are close to normal.

Settings come from configs/review_v1_scans.yaml, flags go on top. Look at the plan first, it calls nothing:

    uv run python scripts/run_review_v1_scans.py --config configs/review_v1_scans.yaml --dry-run true

Then run it. A scan that is done is skipped, and a stopped run resumes where it left off:

    uv run --env-file .env python scripts/run_review_v1_scans.py --config configs/review_v1_scans.yaml

Each scan's outputs go to work/scans/<series uid>/flex_step4/, with status.json saying done, failed or skipped.
Needs PORTKEY_API_KEY for the model calls, or OPENAI_API_KEY with agent.provider set to openai, which --env-file .env provides.
"""

import sys

from fissure_annotation.experiment import model_client
from fissure_annotation.scan_list import (
    ScanListJob,
    plan_lines,
    read_series_uids,
    run_scan_list,
)
from fissure_annotation.settings import load_settings

REVIEW_V1_NEW_SCANS = [
    "1.2.392.200036.9116.2.6.1.20715.4132512272.1761258688.809039",  # emphysema, bullous
    "1.2.392.200036.9116.2.6.1.3268.2047634502.1782770430.932180",  # emphysema, moderate
    "1.2.392.200036.9116.2.6.1.3268.2060189478.1757458418.76856",  # emphysema, mild
    "1.2.392.200036.9116.2.6.1.37.2430222326.1776993777.126945",  # nodule, 10 to 30 mm
    "1.2.392.200036.9116.2.6.1.1236.3294397616.1784072961.125572",  # nodule, 6 to 10 mm
    "1.2.392.200036.9116.2.6.1.20715.4132512272.1772486974.563738",  # septal thickening, fibrotic
    "1.2.392.200036.9116.2.6.1.3268.2047634502.1787960638.28402",  # bronchiectasis, extensive
    "1.2.392.200036.9116.2.6.1.3268.2047634502.1784117731.840568",  # devices, high artifact
    "1.2.392.200036.9116.2.6.1.3268.2056359838.1785648182.659399",  # heart enlargement, the control
]


def require_in_csv(job: ScanListJob) -> None:
    "the reports for the review come from the CSV, so every series must be in it"
    missing = set(REVIEW_V1_NEW_SCANS) - set(read_series_uids(job.csv, job.uid_column))
    if missing:
        raise SystemExit(f"not in {job.csv}: {sorted(missing)}")


if __name__ == "__main__":
    job = load_settings(ScanListJob, sys.argv[1:])
    require_in_csv(job)
    if job.dry_run:
        print("\n".join(plan_lines(job, REVIEW_V1_NEW_SCANS)))
        sys.exit(0)
    statuses = run_scan_list(job, model_client(job.agent), REVIEW_V1_NEW_SCANS)
    failed = [uid for uid, status in statuses.items() if status.state == "failed"]
    print(f"ran {len(statuses)} scans, {len(failed)} failed, see status.json in their folders")
    sys.exit(1 if failed else 0)
