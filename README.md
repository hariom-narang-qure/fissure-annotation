# fissure-annotation

Prototype: a vision model (GPT-6.1-Sol via Portkey) finds lung fissures on CT slices, with a classical ridge tracer and lobe-segmentation hints as helpers. Research only, not diagnostic. Findings, caveats and every experiment so far are in `NOTES.md`.

Run everything with `uv run --env-file <env file with PORTKEY_API_KEY> ...` (see `.env.example`).

## Layout

- `src/fissure_annotation/slicing.py`: cut an isotropic, radiological-orientation lung-window slice (plus lobe labels) from a NIfTI scan.
- `variants.py`, `experiment.py`: one-shot prompt variants A-E (facts, crop, lobe hints, mistake rules) and the blinded-review panels.
- `tracing.py`: `snap`, the ridge tracer (adapted from the exploratory reference code).
- `agent.py`: tool-using slice agent (zoom views, snap, overlay, submit) on the OpenAI Responses API.
- `scroll_agent.py`: the scrolling slice agent (`Stack`, `run_scroll`). `scroll_experiment.py` runs it over the picked slices of a set of scans.
- `config.py`, `settings.py`: `AgentConfig` / `RunConfig`, and the YAML-plus-flags loader.
- `configs/`: one YAML per experiment. `scripts/`: thin entry points.
- `scan_agent.py`: per-scan agent that opens slices, traces, and reports presence for each fissure.
- `review_pack.py`: run the slice agent on a few slices of one scan and write one side-by-side image per slice.
- `prompts/`: the prompts sent to the model.

## Examples

    uv run --env-file .env python -m fissure_annotation.review_pack <scan.nii.gz> <lobe_mask_dir> review/out
    uv run --env-file .env python scripts/run_scroll.py --config configs/scroll_current.yaml
    uv run --env-file .env python scripts/run_scroll.py --config configs/scroll_smoke.yaml     # two slices, for a quick check
    uv run --env-file .env python -m fissure_annotation.scan_agent work/cases.json
    uv run --env-file .env python -m fissure_annotation.experiment work/cases.json E

Lobe masks come from the `total_segmentator` app in muse_tools (`scripts/lobe_crops_for_salvo.py --model total_segmentator --no-crop`).
