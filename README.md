# fissure-annotation

Prototype: a vision model picks waypoints along lung fissures on CT slices, and a classical ridge tracer snaps the line between them. Research only, not diagnostic.

This branch strips the earlier pipeline (lobe-mask hints, scrolling agent, reviewer, one-shot variants).
The git history keeps it.

Run everything with `uv run --env-file <env file with PORTKEY_API_KEY> ...` (see `.env.example`).
