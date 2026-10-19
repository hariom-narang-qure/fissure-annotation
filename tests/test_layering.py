"""CLAUDE.md says layers depend downward only. The imaging layer is the bottom one: it must not know about models or the rest of the package."""

import ast
from pathlib import Path

IMAGING = Path(__file__).parents[1] / "src" / "fissure_annotation" / "imaging"
NOT_ALLOWED_IN_IMAGING = {"openai", "portkey_ai", "pydantic", "pydantic_settings", "yaml"}


def imported_modules(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
    return found


def test_imaging_imports_nothing_from_the_rest_of_the_package():
    for path in IMAGING.glob("*.py"):
        outside = {
            m
            for m in imported_modules(path)
            if m.startswith("fissure_annotation")
            and not m.startswith("fissure_annotation.imaging")
        }
        assert not outside, f"{path.name} imports {sorted(outside)}"


def test_imaging_knows_nothing_about_models_or_config():
    for path in IMAGING.glob("*.py"):
        bad = {m.split(".")[0] for m in imported_modules(path)} & NOT_ALLOWED_IN_IMAGING
        assert not bad, f"{path.name} imports {sorted(bad)}"
