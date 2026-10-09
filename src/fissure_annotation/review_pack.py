"""Run the slice agent on a few slices of one scan and write one image per slice: clean crop | crop with the agent's lines.

Slices: for each plane, the slice with the most lobe-lobe boundary pixels inside each quarter of the lung's extent,
so the review covers the whole lung and every picked slice has a fissure for the agent to find.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fissure_annotation.agent import MODEL, run_agent
from fissure_annotation.experiment import draw_model_lines
from fissure_annotation.picker import best_in
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


def side_by_side(clean: Image.Image, outlined: Image.Image, title: str) -> Image.Image:
    gap, bar = 8, 20
    out = Image.new("RGB", (clean.width + outlined.width + gap, clean.height + bar), "black")
    out.paste(clean, (0, bar))
    out.paste(outlined, (clean.width + gap, bar))
    ImageDraw.Draw(out).text((4, 4), title, fill="white")
    return out


def run_slice(case, plane: str, idx: int, out_dir: Path, model: str) -> dict:
    r = run_agent(cut(case, plane, idx), model, out_dir / "logs" / f"{plane}_{idx:04d}")
    sl = r["slice"]
    n = len(r["fissures"])
    side_by_side(sl.main, draw_model_lines(sl.main, r["fissures"]),
                 f"{plane} slice {idx}: left = lung crop, right = agent's lines ({n}); red = clear, yellow = faint").save(out_dir / f"{plane}_{idx:04d}.png")
    return {"slice": f"{plane}_{idx:04d}", "fissures": [{k: f[k] for k in ("name", "side", "confidence", "evidence", "geometry")} for f in r["fissures"]],
            "steps": r["steps"]}


def main():
    "review_pack <scan.nii.gz> <lobe_dir> <out_dir> [model]"
    scan, lobe_dir, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    model = sys.argv[4] if len(sys.argv) > 4 else MODEL
    out_dir.mkdir(parents=True, exist_ok=True)
    case = load_case(scan, lobe_dir)
    picks = quarter_picks(case)
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda p: run_slice(case, p[0], p[1], out_dir, model), picks))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
