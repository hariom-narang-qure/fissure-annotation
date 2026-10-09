"""Scrolling slice agent restricted to one lung side: the images are cropped to that lung, the lobe hints, the lobe legend and
the ridge tracer see that lung only, and every tool requires both `slice` and `side`.

The run's settings live in a `RunConfig` (plane, start slice, allowed sides, model, limits). `side` is validated against
`config.sides`, so a run can later be allowed several sides; each side has its own crop box and its own views. Line ids are
global across slices. Each tool can also return the completed slice as submitted (with or without lines).
"""
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from portkey_ai import Portkey

from fissure_annotation.agent import FULL, TOOLS, Slice, converse, image, text
from fissure_annotation.config import MODEL, RunConfig
from fissure_annotation.experiment import OUT, draw_model_lines, panel, proxy_scores
from fissure_annotation.slicing import SIDE_LABELS, cut, load_case, side_view
from fissure_annotation.variants import boundary_polylines, facts, hint_image, lung_box, prepare

PROMPT = (Path(__file__).parents[2] / "prompts" / "fissure_side_agent.txt").read_text()
AXIS = {"coronal": 1, "axial": 2}
BOX_STRIDE_MM = 8
MIN_SIDE_VOXELS = 300
SIDE_PROP = {"type": "string", "enum": ["right", "left"], "description": "lung side to act on (required); rejected unless it is a side this run covers"}
SCROLL_PROPS = {
    "slice": {"type": "integer", "description": "slice number to act on (required); must be within the lung range"},
    "completed": {"type": "string", "enum": ["none", "with_lines", "clean"],
                  "description": "also return the completed image of this slice as it would be submitted: with all its traced lines, or clean; omit for none"}}
LOOK = {"type": "function", "name": "look", "description": "Scroll: view the full slice `slice` of lung `side`, clean, with tick marks (a view of that lung's common frame).",
        "parameters": {"type": "object", "properties": {**SCROLL_PROPS, "side": SIDE_PROP, "lobes": {"type": "boolean", "description": "also draw approximate lobe boundaries with a colour legend"}},
                       "required": ["slice", "side"]}}
GUIDE = {
    "right": """- Oblique (major) fissure: separates the right lower lobe from the right upper and middle lobes. Horizontal (minor) fissure: separates the right upper lobe from the right middle lobe.
- Coronal: the horizontal fissure is a near-horizontal line; the oblique fissure is seen obliquely and is often hazy.
- Axial: the oblique fissure is a diagonal line from the mediastinal side outward and posteriorly, but near the hilum it can look nearly horizontal or curved; follow the slope you see. The horizontal fissure usually appears only as an avascular band, rarely as a crisp line.
- Variants exist (accessory or incomplete fissures).""",
    "left": """- Only one fissure: the oblique (major) fissure, separating the left lower lobe from the left upper lobe. The left lung has no horizontal fissure: never trace one.
- Coronal: the oblique fissure is seen obliquely and is often hazy or faint.
- Axial: a diagonal line from the mediastinal side outward and posteriorly; near the hilum it can look nearly horizontal or curved; follow the slope you see.
- Variants exist (accessory or incomplete fissures)."""}
SIDES_ON_IMAGE = {"right": ("right", "left"), "left": ("left", "right")}      # (medial, lateral) edge of the crop: radiological display puts the right lung on the image's left, so its mediastinum is on the crop's right


def side_tools() -> list[dict]:
    tools = [LOOK]
    for t in TOOLS:
        if t["name"] in ("zoom", "snap", "overlay"):
            t = copy.deepcopy(t)
            props = t["parameters"]["properties"]
            props.update(SCROLL_PROPS)
            props["side"] = SIDE_PROP
            t["parameters"]["required"] = ["slice", "side", *[r for r in t["parameters"].get("required", []) if r != "side"]]
        tools.append(t)
    return tools


class Stack:
    "all slices of one plane of one scan for the sides a run covers: built on demand, sharing a crop box and views per side, and line ids"

    def __init__(self, case, cfg: RunConfig):
        self.case, self.cfg, self.plane = case, cfg, cfg.plane
        self.mm_step = float(case.spacing[AXIS[cfg.plane]])         # RAS axes x, y, z = 0, 1, 2
        self.range = {s: self.side_range(s) for s in cfg.sides}
        self.box = {s: self.shared_box(s) for s in cfg.sides}
        self.views = {s: {FULL: {"box": (0.0, 0.0, 1000.0, 1000.0), "contrast": "normal"}} for s in cfg.sides}
        self.slices: dict[tuple[str, int], Slice] = {}
        self.last_line = 0
        self.owner: dict[int, tuple[str, int]] = {}

    def side_range(self, side: str) -> tuple[int, int]:
        axis = AXIS[self.plane]
        in_side = np.isin(self.case.lobes, SIDE_LABELS[side])
        present = np.nonzero(in_side.sum(axis=tuple(i for i in range(3) if i != axis)) > MIN_SIDE_VOXELS)[0]
        return int(present[0]), int(present[-1])

    def shared_box(self, side: str):
        "union of the lung boxes of one lung over slices sampled every few mm, so every slice is cropped the same way"
        lo, hi = self.range[side]
        boxes = []
        for i in range(lo, hi + 1, max(int(BOX_STRIDE_MM / self.mm_step), 1)):
            try:
                boxes.append(lung_box(side_view(cut(self.case, self.plane, i), side)))
            except ValueError:                     # a sampled slice with no usable lung of this side
                continue
        return tuple(int(v) for v in (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)))

    def new_line_id(self) -> int:
        self.last_line += 1
        return self.last_line

    def get(self, side: str, index: int) -> Slice:
        if (side, index) not in self.slices:
            sl = Slice(side_view(cut(self.case, self.plane, index), side), box=self.box[side])
            sl.views, sl.next_id = self.views[side], self.new_line_id
            self.slices[(side, index)] = sl
        return self.slices[(side, index)]

    def validate(self, side, index) -> tuple[str | None, int | None, str]:
        "reject a side this run does not cover, or a slice outside that lung's range, saying what is valid; nothing is done then"
        if side not in self.cfg.sides:
            return None, None, f"side {side!r} is not part of this run; valid sides: {list(self.cfg.sides)}. Nothing was done."
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None, None, f"slice must be an integer; valid slices for the {side} lung are {self.range[side][0]}-{self.range[side][1]}"
        lo, hi = self.range[side]
        if not lo <= index <= hi:
            return None, None, f"slice {index} is outside the {side} lung range {lo}-{hi} for this {self.plane} stack; nearest valid slice is {min(max(index, lo), hi)}. Nothing was done."
        return side, index, ""

    def completed(self, side: str, index: int, mode: str):
        sl = self.get(side, index)
        img, _ = sl.render(FULL)
        n = len(sl.lines)
        if mode == "with_lines":
            sl.draw_lines(img, FULL, sorted(sl.lines))
        return [text(f"completed image of {side} lung, slice {index}, " + (f"with its {n} traced line(s): {sorted(sl.lines)}" if mode == "with_lines" else "clean")), image(img)]

    def call(self, name: str, args: dict, log_dir: Path, step: int, call_id: str):
        "run one tool on the slice and side it names; returns the tool output content"
        args = dict(args)
        side, index, err = self.validate(args.get("side"), args.pop("slice", None))
        mode = args.pop("completed", "none")
        if err:
            return [text(err)]
        if name == "snap" and side == "left" and args.get("name") == "horizontal":
            return [text("the left lung has no horizontal fissure; nothing was traced. Use name oblique (or accessory) if you see another line.")]
        if name != "snap":
            args.pop("side", None)
        sl = self.get(side, index)
        if name == "look":
            img, legend = sl.render(FULL, args.get("lobes", False))
            content = [text(f"{side} lung, slice {index}: full view (ticks drawn inside the edges)" + ("; lobe boundaries: " + "; ".join(legend) if legend else "")), image(img)]
        else:
            content, img = {"snap": sl.snap_tool, "overlay": sl.overlay_tool, "zoom": sl.zoom_tool}[name](**args)
            content[0]["text"] = f"[{side} lung, slice {index}] " + content[0]["text"]
        if name == "snap":
            for lid in sl.lines:
                self.owner[lid] = (side, index)
        if img is not None:
            img.save(log_dir / f"step{step:02d}_{name}_{side[0]}{index}_{call_id[-6:]}.png")
        if mode != "none":
            content = content + self.completed(side, index, mode)
        return content

    def final_by_slice(self, submitted: list[dict]) -> dict[tuple[str, int], list[dict]]:
        out: dict[tuple[str, int], list[dict]] = {}
        for s in submitted:
            key = self.owner.get(s["line_id"])
            if key is not None:
                out.setdefault(key, []).extend(self.slices[key].final_fissures([s]))
        return out


def initial_input(stack: Stack) -> list[dict]:
    content = []
    for side in stack.cfg.sides:
        sl = stack.get(side, stack.cfg.start)
        p = sl.prepared
        hint = hint_image(sl.view, p.box, boundary_polylines(sl.view, p.box, p.scale), p.scale)
        content += [text(f"{side.capitalize()} lung, slice {stack.cfg.start}: image 1 is the slice, image 2 has the approximate lobe boundaries."), image(sl.main), image(hint)]
    return [{"role": "user", "content": [text(f"Find and trace the fissures of the {' and '.join(stack.cfg.sides)} lung in slice {stack.cfg.start}. You can scroll to other slices with the tools.")] + content}]


def run_side(case, cfg: RunConfig, log_dir: Path) -> dict:
    if len(cfg.sides) != 1:
        raise ValueError("the prompt is written for one side per run; use one-element sides")
    side = cfg.sides[0]
    log_dir.mkdir(parents=True, exist_ok=True)
    client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    stack = Stack(case, cfg)
    lo, hi = stack.range[side]
    if not lo <= cfg.start <= hi:
        raise ValueError(f"start slice {cfg.start} outside the {side} lung range {lo}-{hi}")
    medial, lateral = SIDES_ON_IMAGE[side]          # which side of the crop the mediastinum is on, and the chest wall
    instructions = PROMPT.format(start=cfg.start, plane=cfg.plane, top=facts(stack.get(side, cfg.start).view)["top"], lo=lo, hi=hi, mm_step=stack.mm_step,
                                 five_mm=max(round(5 / stack.mm_step), 1), side=side, medial=medial, lateral=lateral,
                                 guide=GUIDE[side])
    kw = dict(model=cfg.model, instructions=instructions, tools=side_tools() + [t for t in TOOLS if t["name"] == "submit"],
              reasoning=cfg.reasoning, max_output_tokens=cfg.max_output_tokens)
    finished, transcript, usage, steps = converse(client, kw, initial_input(stack),
                                                  lambda c, k: stack.call(c.name, json.loads(c.arguments), log_dir, k, c.call_id), "submit", cfg.max_steps)
    submitted = finished["lines"] if finished else [{"line_id": i, "status": "uncertain", "geometry": "mine", "evidence": "agent did not submit"} for i in stack.owner]
    by_slice = stack.final_by_slice(submitted)
    (log_dir / "transcript.json").write_text(json.dumps({"transcript": transcript, "submitted": submitted, "usage": usage}, indent=1))
    visited = sorted({c["args"].get("slice") for s in transcript for c in s["calls"] if c["args"].get("slice") is not None})
    rejected = sum(1 for s in transcript for res in s.get("results", []) for r in res if "Nothing was done" in r or "no horizontal fissure" in r)
    return {"fissures": by_slice.get((side, cfg.start), []), "extra": {k: v for k, v in by_slice.items() if k != (side, cfg.start)}, "steps": steps, "usage": usage,
            "visited": visited, "rejected_calls": rejected, "stack": stack, "side": side, "submitted_explicitly": finished is not None}


def reframe(wp: dict, sl: Slice, ref) -> dict:
    "a waypoint in one lung's crop frame -> the full-lung crop frame used by the comparison panels"
    col, row = sl.to_px(wp["x"], wp["y"])
    c0, r0, c1, r1 = ref.box
    return {"x": round((col - c0) / (c1 - c0) * 1000), "y": round((row - r0) / (r1 - r0) * 1000)}


def run_slice_both_sides(case, name: str, plane: str, idx: int, model: str) -> dict:
    "run a right-lung agent and a left-lung agent on the same slice and merge their lines onto the usual full-lung panel"
    d = OUT / name / f"{plane}_{idx}"
    cfgs = {side: RunConfig(plane=plane, start=idx, sides=(side,), model=model) for side in ("right", "left")}
    try:
        with ThreadPoolExecutor(2) as ex:
            runs = dict(zip(cfgs, ex.map(lambda s: run_side(case, cfgs[s], d / f"agent_log_side_{s}"), cfgs)))
    except Exception as e:
        return {"case": name, "plane": plane, "index": idx, "variant": "AGSIDE", "error": str(e)[:300]}
    view = cut(case, plane, idx)
    ref = prepare(view, "E")
    merged = []
    for side, r in runs.items():
        sl = r["stack"].get(side, idx)
        merged += [{**f, "waypoints": [reframe(w, sl, ref) for w in f["waypoints"]]} for f in r["fissures"]]
    (d / "AGSIDE.json").write_text(json.dumps(merged, indent=2))
    polys = boundary_polylines(view, ref.box, ref.scale)
    panel(ref, hint_image(view, ref.box, polys, ref.scale), draw_model_lines(ref.images[0], merged)).save(d / "AGSIDE.panel.png")
    return {"case": name, "plane": plane, "index": idx, "variant": "AGSIDE", "n_fissures": len(merged),
            "per_side": {s: {"steps": r["steps"], "tokens": r["usage"], "visited": r["visited"], "rejected_calls": r["rejected_calls"], "n": len(r["fissures"])} for s, r in runs.items()},
            **proxy_scores(view, merged, ref)}


def main():
    "side_agent <cases.json> [model] [only_case_substring]"
    cases = json.load(open(sys.argv[1]))
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    only = sys.argv[3] if len(sys.argv) > 3 else ""
    jobs = []
    for name, (scan, lobe_dir) in cases.items():
        if only not in name:
            continue
        case = load_case(scan, lobe_dir)
        jobs += [(case, name, plane, idx) for plane, idx in json.load(open(OUT / name / "picks.json"))]
    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(lambda j: run_slice_both_sides(*j, model), jobs))
    (OUT / "results_side.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
