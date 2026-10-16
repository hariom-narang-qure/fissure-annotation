# Prose guidelines

These apply to code, docs, READMEs, PRs and commits.

- Do not compose multiple sentences into one.
- Do not use em dashes. They tend to join sentences that should be separate.
- Use bullet points generally.
  - Use nested bullets to continue the flow of the previous point.
- Be concise, but prefer simplicity over conciseness.

# Coding and documentation guidelines

- Prefer small readable functions which do one thing.
  - Extract a function to make code readable even if it is not reused.
- Convey meaning with descriptive function and variable names.
  - Do not write a docstring that restates the signature.
  - Err on the side of not documenting. Types and names carry intent.
- Docstrings explain the why and how of non-trivial parts.
  - Document algorithm steps instead of writing long strings.
  - For ingestion or cleaning code, say what was done, why, and give references.
- Keep the code dumb and boring. Comment anything that feels implicit.
- Prefer a class when several functions repeatedly pass the same large set of arguments.
- Prefer strong typing with descriptive type names.
  - Pydantic is preferred over dataclasses, since it gives JSON marshalling.
  - Use jaxtyping for array shapes.
- Do not use matplotlib for non-interactive work. Composite with PIL.

# Project structure

- The library in `src/fissure_annotation/` holds all logic.
- Experiments are thin wrappers on top of the library.
  - An experiment is a YAML in `configs/` plus a small script in `scripts/`.
  - Anything that distinguishes one experiment from another belongs in git, in the config.
  - A script only decides what the inputs are and which pieces to assemble.
- Extension points are small interfaces, not flags.
  - A new agent variant is a new tool set, prompt and result mapper.
  - Do not thread boolean flags through the pipeline.
- Layers depend downward only: imaging, waypoint storage, tools, conversation, experiments.
  - `imaging/` is the bottom layer: it handles pixels and nothing else, and imports nothing from the rest of the package.
  - A test in `tests/test_layering.py` enforces it.
- Tests live in `tests/`. A behaviour-preserving refactor needs a test that pins the behaviour first.
