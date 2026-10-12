# fissure-annotation

Prototype: a vision model (GPT-6.1-Sol via Portkey) finds lung fissures on CT slices, with a classical ridge tracer and lobe-segmentation hints as helpers. Research only, not diagnostic. Findings, caveats and every experiment so far are in `NOTES.md`.

Run everything with `uv run --env-file <env file with PORTKEY_API_KEY> ...` (see `.env.example`).

## Layout

The library holds all logic. Experiments are a YAML in `configs/` plus a thin script in `scripts/`.

Layers depend downward only:

- `slicing.py`, `picker.py`: cut an isotropic, radiological-orientation lung-window slice (plus lobe labels) from a NIfTI scan, and pick slices.
- `variants.py`, `client.py`: what the one-shot prompts show the model (variants A to E) and the one-shot call.
- `tracing.py`: `snap`, the ridge tracer.
- `frames.py`: the three coordinate frames (slice pixels, main 0-1000 frame, view frame) and the maps between them.
- `drawing/`: every function that puts pixels on an image. Pure PIL, no slice or frame knowledge.
- `lines.py`: pydantic models for lines, submitted lines and fissures.
- `tools/`: the tool schemas (`schemas.py`) and the tool lists built for each agent setup (`toolsets.py`).
- `agent.py`: `Slice` (frames, rendering, the tool implementations) and `converse`, the model conversation loop.
- `prompts.py`: builds the scroll agent instructions from the template files in `prompts/`.
- `scroll_agent.py`, `scan_agent.py`, `reviewer.py`: the scrolling slice agent (`Stack`), the per-scan agent, and the separate reviewer call.
- `evaluation/`: proxy scores and review panels. Nothing here runs a model.
- `config.py`, `settings.py`, `cases.py`: `AgentConfig` and `RunConfig`, the YAML-plus-flags loader, and which scans and slices a run covers.
- `scroll_experiment.py`, `scan_experiment.py`, `slice_experiment.py`, `experiment.py`: run the above over the picked slices, and write the outputs.
- `review_pack.py`: run an agent on a few slices of one scan and write one side-by-side image per slice.
- `__init__.py`, `overlay.py`, `client.propose_fissures`: the original GLM one-shot command line (`fissure-annotation <image>`). Legacy, not used by any experiment.

## Examples

Every experiment is a YAML in `configs/` and a thin script in `scripts/`. Flags go on top of the YAML.

    uv run --env-file .env python scripts/run_scroll.py --config configs/scroll_current.yaml
    uv run --env-file .env python scripts/run_scroll.py --config configs/scroll_smoke.yaml        # two slices, a quick check
    uv run --env-file .env python scripts/run_scroll.py --config configs/scroll_current.yaml --agent.plain-images true
    uv run --env-file .env python scripts/run_scan.py --config configs/scan_agent.yaml
    uv run --env-file .env python scripts/run_one_shot.py --config configs/one_shot_e.yaml
    uv run --env-file .env python scripts/run_slice_agent.py --config configs/slice_agent.yaml
    uv run --env-file .env python -m fissure_annotation.review_pack <scan.nii.gz> <lobe_mask_dir> review/out

Lobe masks come from the `total_segmentator` app in muse_tools (`scripts/lobe_crops_for_salvo.py --model total_segmentator --no-crop`).
