"""CLAUDE.md says layers depend downward only. This keeps the lower layers from importing the upper ones again."""
import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "fissure_annotation"
AGENTS = {"agent", "scroll_agent", "scan_agent", "reviewer"}
RUNNERS = {"experiment", "scroll_experiment", "scan_experiment", "slice_experiment", "review_pack"}

FORBIDDEN = {
    "drawing": AGENTS | RUNNERS | {"variants", "evaluation", "tools", "prompts", "cases"},
    "evaluation": AGENTS | RUNNERS | {"tools", "prompts", "cases"},
    "tools": AGENTS | RUNNERS | {"evaluation", "drawing"},
    "frames": AGENTS | RUNNERS | {"drawing", "evaluation", "tools", "variants"},
    "lines": AGENTS | RUNNERS | {"drawing", "evaluation", "tools", "variants"},
    "slicing": AGENTS | RUNNERS | {"drawing", "evaluation", "tools", "variants", "lines"},
    "tracing": AGENTS | RUNNERS | {"drawing", "evaluation", "tools", "variants", "lines", "slicing"},
    "agent": RUNNERS | {"evaluation", "cases"},
    "scroll_agent": RUNNERS | {"evaluation", "cases"},
    "scan_agent": {"experiment", "scroll_experiment", "slice_experiment", "cases"},
}


def imported_modules(path: Path) -> set[str]:
    "the second part of every `fissure_annotation.<x>...` import in the file"
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node, ast.Import) else []
        found |= {n.split(".")[1] for n in names if n.startswith("fissure_annotation.") and len(n.split(".")) > 1}
    return found


def files_of(layer: str) -> list[Path]:
    package = PACKAGE / layer
    return sorted(package.rglob("*.py")) if package.is_dir() else [PACKAGE / f"{layer}.py"]


@pytest.mark.parametrize("layer", FORBIDDEN)
def test_a_layer_does_not_import_the_layers_above_it(layer):
    for path in files_of(layer):
        bad = imported_modules(path) & FORBIDDEN[layer]
        assert not bad, f"{path.relative_to(PACKAGE)} imports {sorted(bad)}"
