# fissure-annotation

Prototype: a vision model (GPT-6.1-Sol via Portkey) picks waypoints along lung fissures on axial CT slices, and a classical ridge tracer joins them.
Research only, not diagnostic.

## How it works

- `imaging/` cuts a NIfTI scan into smoothed, lung-windowed axial PNGs and traces a ridge path between waypoints.
  - It imports nothing else from the package, and a test keeps it that way.
- `agent.py` runs one model conversation per slice with three tools: `zoom` (a crop with pixel ticks), `trace_points` and `finish`.
  - The waypoints are saved to `points/` before they are traced, so they can be reloaded later.
- `experiment.py` runs the agent over a range of slices in parallel and writes a contact sheet.

## Run

    uv run python scripts/make_slices.py --config configs/make_slices.yaml
    uv run --env-file .env python scripts/run_waypoint_agent.py --config configs/waypoint_agent.yaml

- `.env` needs `PORTKEY_API_KEY` (see `.env.example`).
- Flags go on top of the YAML, for example `--agent.max-steps 20` or `--first-slice 160 --last-slice 170`.
- Output goes under `out_dir`: `points/` (the model's waypoints), `masks/`, `overlays/`, `logs/` and `contact_sheet.png`.

## Fissure mask for a scan

    uv run --env-file .env python scripts/make_fissure_mask.py --config configs/make_fissure_mask.yaml \
        --nifti scan.nii.gz --mask-path out/scan_fissures.nii.gz --first-slice 165 --last-slice 179

- The mask is a 0/1 `uint8` NIfTI with the same shape and affine as the scan, one voxel wide in plane and one slice thick.
- The agent calls the model once per slice. Leave out `--first-slice` and `--last-slice` to run every slice, which costs roughly 15k input tokens per slice.
- The slice PNGs and the agent's points, masks and logs go to `<mask name>_work/` next to the mask, or to `--work-dir`.
- Running the same command again skips slices that already have a log, so a run that failed halfway can be finished.

## Earlier pipeline

The lobe-mask scroll agent, reviewer and one-shot variants were removed.
The git history before the "Strip the lobe-mask scroll pipeline" commit has them, with their notes.
