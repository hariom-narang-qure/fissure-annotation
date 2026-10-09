"""Run variants A-D of the v2 prompt with one model over a set of slices; save json, raw reply and a review panel each.

Proxy score (NOT accuracy): the lobe-segmentation boundaries stand in for fissures. `dist_mm` is the mean distance
from the model's polyline to the nearest lobe boundary; `within5` the share of polyline points within 5 mm;
`recall5` the share of long boundary pixels within 5 mm of any model polyline. C and D were shown these boundaries,
so for them the proxy is not independent of the input.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from fissure_annotation.client import ask, parse_json
from fissure_annotation.picker import pick_slices
from fissure_annotation.slicing import boundary_mask, cut, load_case
from fissure_annotation.variants import boundary_polylines, hint_image, prepare, to_slice_pixels

MODEL = "@openai/gpt-6.1-sol"
CONF_COLOUR = {"clear": (255, 40, 40), "faint": (255, 220, 0), "uncertain": (255, 140, 0)}
OUT = Path("work/exp")


def polyline_points(pts: list[tuple[float, float]], step: float = 2.0) -> np.ndarray:
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 1)
        out += [(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n) for t in range(n + 1)]
    return np.array(out)


def proxy_scores(view, fissures, prepared) -> dict:
    bmask = boundary_mask(view.lobes)
    if not bmask.any() or not fissures:
        return {"dist_mm": None, "within5": None, "recall5": None}
    dist_to_boundary = ndimage.distance_transform_edt(~bmask) * view.mm_per_px
    h, w = bmask.shape
    pts = np.vstack([polyline_points([to_slice_pixels(p, prepared) for p in f["waypoints"]]) for f in fissures])
    ix = np.clip(pts.round().astype(int), 0, [w - 1, h - 1])
    d = dist_to_boundary[ix[:, 1], ix[:, 0]]
    line = np.zeros_like(bmask)
    line[ix[:, 1], ix[:, 0]] = True
    dist_to_line = ndimage.distance_transform_edt(~line) * view.mm_per_px
    return {"dist_mm": round(float(d.mean()), 1), "within5": round(float((d <= 5).mean()), 2),
            "recall5": round(float((dist_to_line[bmask] <= 5).mean()), 2)}


def draw_model_lines(img: Image.Image, fissures) -> Image.Image:
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    for k, f in enumerate(fissures, start=1):
        px = [(p["x"] / 1000 * img.width, p["y"] / 1000 * img.height) for p in f["waypoints"]]
        col = CONF_COLOUR.get(f["confidence"], (255, 140, 0))
        d.line(px, fill=col, width=2)
        d.text((px[0][0] + 4, px[0][1] - 12), f"{k} {f['side'][0].upper()} {f['confidence']}", fill=col)
    return out


def panel(prepared, hint: Image.Image, overlay: Image.Image) -> Image.Image:
    tiles = [("1 model input", prepared.images[0].convert("RGB")), ("2 model output", overlay),
             ("3 lobe-segmentation boundaries (imperfect, reference only)", hint)]
    h = max(t.height for _, t in tiles)
    canvas = Image.new("RGB", (sum(t.width for _, t in tiles) + 8 * len(tiles), h + 18), "black")
    x = 0
    for title, t in tiles:
        canvas.paste(t, (x, 18))
        ImageDraw.Draw(canvas).text((x + 4, 3), title, fill="white")
        x += t.width + 8
    return canvas


def run_variant(case: str, plane: str, idx: int, view, variant: str, model: str) -> dict:
    d = OUT / case / f"{plane}_{idx}"
    d.mkdir(parents=True, exist_ok=True)
    prepared = prepare(view, variant)
    for attempt in range(2):
        try:
            reply = ask(prepared.images, prepared.prompt, model=model)
            fissures = parse_json(reply)["fissures"]
            break
        except Exception as e:
            err = str(e)[:200]
    else:
        return {"case": case, "plane": plane, "index": idx, "variant": variant, "error": err}
    (d / f"{variant}.raw.txt").write_text(reply)
    (d / f"{variant}.json").write_text(json.dumps(fissures, indent=2))
    overlay = draw_model_lines(prepared.images[0], fissures)
    polys = boundary_polylines(view, prepared.box, prepared.scale)
    hint = hint_image(view, prepared.box, polys, prepared.scale)
    panel(prepared, hint, overlay).save(d / f"{variant}.panel.png")
    return {"case": case, "plane": plane, "index": idx, "variant": variant, "n_fissures": len(fissures),
            **proxy_scores(view, fissures, prepared)}


def main():
    "experiment <cases.json> <variants e.g. ABCD> [model]; cases.json = {name: [scan, lobe_dir]}"
    cases = json.load(open(sys.argv[1]))
    variants = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL
    jobs = []
    for name, (scan, lobe_dir) in cases.items():
        case = load_case(scan, lobe_dir)
        picks_file = OUT / name / "picks.json"
        picks_file.parent.mkdir(parents=True, exist_ok=True)
        picks = json.load(open(picks_file)) if picks_file.exists() else pick_slices(case)
        json.dump(picks, open(picks_file, "w"))
        for plane, idx in picks:
            view = cut(case, plane, idx)
            jobs += [(name, plane, idx, view, v) for v in variants]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda j: run_variant(*j[:3], j[3], j[4], model), jobs))
    out = OUT / f"results_{variants}_{model.split('/')[-1]}.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
