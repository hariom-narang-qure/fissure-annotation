"""Run the slice agent on a few slices of one scan and write one image per slice: clean crop | crop with the agent's lines.

Slices: for each plane, the slice with the most lobe-lobe boundary pixels inside each quarter of the lung's extent,
so the review covers the whole lung and every picked slice has a fissure for the agent to find.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fissure_annotation.agent import MODEL, run_agent
from fissure_annotation.drawing.model_lines import draw_model_lines
from fissure_annotation.drawing.panels import side_by_side
from fissure_annotation.picker import best_in
from fissure_annotation.config import RunConfig
from fissure_annotation.scroll_agent import run_scroll
from fissure_annotation.slicing import cut, load_case

AXIS = {"axial": 2, "coronal": 1}
QUARTERS = 4


def quarter_picks(case) -> list[tuple[str, int]]:
    picks = []
    for plane, axis in AXIS.items():
        present = np.nonzero((case.lobes > 0).sum(axis=tuple(i for i in range(3) if i != axis)) > 500)[0]
        edges = np.linspace(present[0], present[-1], QUARTERS + 1).astype(int)
        picks += [(plane, best_in(case, plane, int(a), int(b))) for a, b in zip(edges[:-1], edges[1:])]
    return picks


def run_slice(case, plane: str, idx: int, out_dir: Path, model: str, scroll: bool = False, use_tracer: bool = True) -> dict:
    log = out_dir / "logs" / f"{plane}_{idx:04d}"
    if scroll:
        r = run_scroll(case, RunConfig(plane=plane, start=idx, model=model, use_tracer=use_tracer), log)
        sl, visited = r["stack"].get(idx), r["visited"]
        title_extra = f"; scrolled through slices {visited}" + ("" if use_tracer else "; no tracer, points checked with check_points")
    else:
        r = run_agent(cut(case, plane, idx), model, log)
        sl, visited, title_extra = r["slice"], [idx], ""
    n = len(r["fissures"])
    side_by_side(sl.main, draw_model_lines(sl.main, r["fissures"]),
                 f"{plane} slice {idx}: left = lung crop, right = agent's lines ({n}); red = clear, yellow = faint{title_extra}").save(out_dir / f"{plane}_{idx:04d}.png")
    return {"slice": f"{plane}_{idx:04d}", "fissures": [{k: getattr(f, k) for k in ("name", "side", "confidence", "evidence", "geometry")} for f in r["fissures"]],
            "steps": r["steps"], "slices_visited": visited}


def main():
    "review_pack <scan.nii.gz> <lobe_dir> <out_dir> [model] [scroll] [notracer]"
    scan, lobe_dir, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    model = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] not in ("scroll", "notracer") else MODEL
    scroll, use_tracer = "scroll" in sys.argv[4:], "notracer" not in sys.argv[4:]
    out_dir.mkdir(parents=True, exist_ok=True)
    case = load_case(scan, lobe_dir)
    picks = quarter_picks(case)
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda p: run_slice(case, p[0], p[1], out_dir, model, scroll, use_tracer), picks))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
