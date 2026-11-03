# fissure-annotation

Prototype: a vision model (GPT-6.1-Sol via Portkey) picks waypoints along lung fissures on axial, coronal or sagittal CT slices, and a classical ridge tracer joins them.
Research only, not diagnostic.

## How it works

- `imaging/` cuts a NIfTI scan into smoothed, lung-windowed PNGs (axial, coronal or sagittal) and traces a ridge path between waypoints.
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
- `--views` picks the planes the model annotates, any of `axial`, `coronal` and `sagittal` (default `axial`). Repeat the flag or comma-separate them, for example `--views axial --views coronal` or `--views axial,coronal,sagittal`.
- With several views the model annotates every slice of each view, builds one 3D mask per view, and the final mask is their voxel-wise OR.
- Slice numbers count along a view's axis, so `--first-slice` and `--last-slice` need exactly one view.
- Coronal and sagittal slices are cut on the voxel grid, so they look stretched along z when the slice spacing differs from the in-plane spacing.
- The agent annotates the slices in stacks of 5 nearby slices, one conversation per stack. Leave out `--first-slice` and `--last-slice` to run every slice.
- The slice PNGs and the agent's points, masks and logs go to `<mask name>_work/<view>/` next to the mask, or to `--work-dir`.
- Running the same command again skips stacks that already have a log, so a run that failed halfway can be finished.

## Scan pipeline

    uv run --env-file .env python scripts/run_scan_list.py --config configs/pipeline_terra_ridge_step3.yaml --csv <scans.csv>

- This is the current setup: gpt-6.1-terra, stacks of 5 slices on the ridge map, slices 3 apart, all three views.
- Each scan gets `work/scans/<uid>/<run_name>/`, with the sheet in `post/sheet.nii.gz`.
- `--dry-run true` prints the slices and an estimated cost, and calls nothing.
- For a long list, run it detached, so it outlives the login, and add `--agent.service-tier flex` for the cheaper tier:

      tmux new -d -s fissure 'uv run --env-file .env python scripts/run_scan_list.py --config configs/pipeline_terra_ridge_step3.yaml --csv <scans.csv> --agent.service-tier flex'

- What happens on an error is in `src/fissure_annotation/retry.py` (per request) and `scan_list.py` step 7 (per scan).
- The experiments that led to this setup are in `docs/stack_agent_experiments.md`.

## Earlier pipeline

The lobe-mask scroll agent, reviewer and one-shot variants were removed.
The git history before the "Strip the lobe-mask scroll pipeline" commit has them, with their notes.
