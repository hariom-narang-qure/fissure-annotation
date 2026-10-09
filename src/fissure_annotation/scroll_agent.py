"""Slice agent that can scroll: every tool takes the slice number to act on (required), in one plane of one scan.

The model starts on a given slice (named in the prompt) and may look at, zoom into, trace on and compare any other slice
of the same plane to decide whether a line is really a fissure; the lines on the start slice are the deliverable. All
slices of a plane share one crop box (the union of their lung boxes), so a 0-1000 position, and a zoomed view, mean the
same anatomy on every slice. Line ids are global. Every tool can also return the completed slice as submitted, with
or without the lines drawn on it.
"""
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from portkey_ai import Portkey

from fissure_annotation.agent import FULL, TOOLS, Slice, converse, image, text
from fissure_annotation.config import MODEL, RunConfig
from fissure_annotation.reviewer import Reviewer
from fissure_annotation.experiment import OUT, draw_model_lines, panel, proxy_scores
from fissure_annotation.slicing import cut, load_case
from fissure_annotation.variants import boundary_polylines, facts, hint_image, lung_box

PROMPTS = {True: (Path(__file__).parents[2] / "prompts" / "fissure_scroll_agent.txt").read_text(),
           False: (Path(__file__).parents[2] / "prompts" / "fissure_scroll_agent_notracer.txt").read_text()}   # keyed by RunConfig.use_tracer
AXIS = {"coronal": 1, "axial": 2}
BOX_STRIDE_MM = 8
SCROLL_PROPS = {
    "slice": {"type": "integer", "description": "slice number to act on (required); must be within the lung range"},
    "completed": {"type": "string", "enum": ["none", "with_lines", "clean"],
                  "description": "also return the completed image of this slice as it would be submitted: with all its traced lines, or clean; omit for none"}}
LOOK = {"type": "function", "name": "look", "description": "Scroll: view slice `slice`, clean, with tick marks, in the view `view_id` (default: the full slice). Pass the view you are working in to see a neighbouring slice at exactly the same zoom and region.",
        "parameters": {"type": "object", "properties": {**SCROLL_PROPS, "view_id": {"type": "string", "description": "'full' or a zoom id; the slice is shown in that view's region, with that view's contrast"},
                                                          "lobes": {"type": "boolean", "description": "also draw approximate lobe boundaries with a colour legend"}},
                       "required": ["slice"]}}


REVIEW_PARAGRAPH = ("\n\nReview (optional): `overlay(..., review: true)` also asks a separate, independent reviewer to check your lines against a rule list (ends, no interpolation, points on the line, smooth path, no duplicates, right structure, naming, missed fissures) and to tell you which lines, and which numbered points of them, look problematic and why, in words only (no coordinates). "
                    "Use it once when your lines for the slice are drawn, before you submit; you have at most {max_reviews} reviews per run. The reviewer can be wrong: verify each finding by zooming on that spot before you change anything, and keep a line you can see is right.")


CHECK_POINTS = {"type": "function", "name": "check_points",
                "description": ("Preview where waypoints land, before you commit them: shows the clean view next to the same view with your points as numbered rings. Nothing is stored. "
                                "The thin line you mean to follow should pass through the centre of every ring; if a point is off, correct its coordinates and check again. Free to repeat."),
                "parameters": {"type": "object", "properties": {
                    **SCROLL_PROPS,
                    "waypoints": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}},
                    "view_id": {"type": "string", "description": "the view whose image you are looking at ('full' or a zoom id); the points are in its frame"},
                    "connect": {"type": "boolean", "description": "also join the points with a thin line (hides a thin fissure, so off by default)"}},
                    "required": ["slice", "waypoints"]}}


def without_ticks(tools: list[dict]) -> list[dict]:
    "tool descriptions for the plain-PNG mode: nothing on the images to read coordinates from"
    out = copy.deepcopy(tools)
    for t in out:
        t["description"] = (t["description"].replace(" (ticks are drawn inside the image edges, in that frame)", "").replace(", clean, with tick marks,", ", clean,"))
    return out


PLAIN_PROMPT_EDITS = [("; tick marks and labels are drawn inside the edges of zoomed images every 100 units", "; the images have no tick marks or labels, so place points by eye as a fraction of the image width and height (the centre is 500, 500)"),
                      (" clean, with tick marks, in the view", " clean, in the view")]


def scroll_tools(use_tracer: bool = True, use_reviewer: bool = False, plain: bool = False) -> list[dict]:
    tools = _scroll_tools(use_tracer, use_reviewer)
    return without_ticks(tools) if plain else tools


def without_tracer(tools: list[dict]) -> list[dict]:
    "snap -> mark (waypoints joined by straight segments), and submit loses the traced/own geometry choice"
    out = []
    for t in tools:
        t = copy.deepcopy(t)
        if t["name"] == "snap":
            t["name"] = "mark"
            t["description"] = ("Draw a fissure line. Give waypoints along the fissure, ordered from one end to the other, in the frame of `view_id`. The line is your waypoints joined by straight segments, "
                                "so add a point wherever the fissure bends. It is stored as a numbered line in full-image coordinates, and you get back the view image with the line and a circle at each point. "
                                "Pass line_id to redo an existing line with corrected points.")
        if t["name"] == "submit":
            item = t["parameters"]["properties"]["lines"]["items"]
            item["properties"].pop("geometry")
            item["required"] = [r for r in item["required"] if r != "geometry"]
        out.append(t)
    return out


def _scroll_tools(use_tracer: bool = True, use_reviewer: bool = False) -> list[dict]:
    tools = [LOOK, CHECK_POINTS]
    for t in TOOLS:
        if t["name"] in ("zoom", "snap", "overlay"):
            t = copy.deepcopy(t)
            t["parameters"]["properties"].update(SCROLL_PROPS)
            t["parameters"]["required"] = ["slice", *t["parameters"].get("required", [])]
            if t["name"] == "overlay" and use_reviewer:
                t["parameters"]["properties"]["review"] = {"type": "boolean", "description": "also ask a separate reviewer agent to check the lines on this slice against a rule list and report which lines and points look wrong and why (limited number per run)"}
        tools.append(t)
    return tools if use_tracer else without_tracer(tools)


class Stack:
    "all slices of one plane of one scan, built on demand, sharing a crop box, views and line ids"

    def __init__(self, case, plane: str, use_tracer: bool = True, reviewer: Reviewer | None = None, plain: bool = False):
        self.case, self.plane, self.use_tracer, self.reviewer, self.plain = case, plane, use_tracer, reviewer, plain
        axis = AXIS[plane]
        present = np.nonzero((case.lobes > 0).sum(axis=tuple(i for i in range(3) if i != axis)) > 500)[0]
        self.lo, self.hi = int(present[0]), int(present[-1])
        mm = case.spacing[axis]                      # RAS axes: x, y, z = 0, 1, 2
        self.mm_step = float(mm)
        self.box = self.shared_box()
        self.slices: dict[int, Slice] = {}
        self.views = {FULL: {"box": (0.0, 0.0, 1000.0, 1000.0), "contrast": "normal"}}
        self.last_line = 0
        self.owner: dict[int, int] = {}

    def shared_box(self):
        "union of the lung boxes of slices sampled every few mm, so every slice is cropped the same way"
        step = max(int(BOX_STRIDE_MM / self.mm_step), 1)
        boxes = [lung_box(cut(self.case, self.plane, i)) for i in range(self.lo, self.hi + 1, step)]
        return tuple(int(v) for v in (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)))

    def new_line_id(self) -> int:
        self.last_line += 1
        return self.last_line

    def get(self, index: int) -> Slice:
        if index not in self.slices:
            sl = Slice(cut(self.case, self.plane, index), box=self.box, use_tracer=self.use_tracer, plain=self.plain)
            sl.views, sl.next_id = self.views, self.new_line_id
            self.slices[index] = sl
        return self.slices[index]

    def validate(self, index) -> tuple[int | None, str]:
        "reject a slice number outside the lung range, naming the valid range and the nearest valid slice"
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None, f"slice must be an integer; valid slices are {self.lo}-{self.hi}"
        if not self.lo <= index <= self.hi:
            return None, f"slice {index} is outside the lung range {self.lo}-{self.hi} for this {self.plane} stack; nearest valid slice is {min(max(index, self.lo), self.hi)}. Nothing was done."
        return index, ""

    def completed(self, index: int, mode: str, view_id: str = FULL):
        "the slice as submitted (with or without its lines), in the view the call was working in"
        sl = self.get(index)
        img, _ = sl.render(view_id)
        n = len(sl.lines)
        if mode == "with_lines":
            sl.draw_lines(img, view_id, sorted(sl.lines))
        return [text(f"completed image of slice {index} (view {view_id}), " + (f"with its {n} traced line(s): {sorted(sl.lines)}" if mode == "with_lines" else "clean")), image(img)]

    def call(self, name: str, args: dict, log_dir: Path, step: int, call_id: str):
        "run one tool on the slice it names; returns the tool output content"
        args = dict(args)
        index, err = self.validate(args.pop("slice", None))
        mode = args.pop("completed", "none")
        want_review = args.pop("review", False)
        if err:
            return [text(err)]
        sl = self.get(index)
        if name == "look":
            view_id = args.get("view_id", FULL)
            sl.box(view_id)                      # an unknown view id raises a clear error that the model is told about
            img, legend = sl.render(view_id, args.get("lobes", False))
            content = [text(f"slice {index}, view {view_id} ({sl.tick_note()}; the same region on every slice)" + ("; lobe boundaries: " + "; ".join(legend) if legend else "")), image(img)]
        else:
            tools = {"overlay": sl.overlay_tool, "zoom": sl.zoom_tool, "check_points": sl.check_points_tool, **({"snap": sl.snap_tool} if self.use_tracer else {"mark": sl.mark_tool})}
            if name not in tools:
                return [text(f"unknown tool {name!r}; available: {sorted(tools) + ['look', 'submit']}")]
            content, img = tools[name](**args)
            content[0]["text"] = f"[slice {index}] " + content[0]["text"]
            if name == "overlay" and want_review:
                content = content + (self.reviewer.review(sl, index, args.get("line_ids"), log_dir, step) if self.reviewer else [text("review is not available in this run")])
        if name in ("snap", "mark"):
            for lid, line in sl.lines.items():
                self.owner[lid] = index
        if img is not None:
            img.save(log_dir / f"step{step:02d}_{name}_s{index}_{call_id[-6:]}.png")
        if mode != "none":
            content = content + self.completed(index, mode, args.get("view_id", FULL))
        return content

    def final_by_slice(self, submitted: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for s in submitted:
            idx = self.owner.get(s["line_id"])
            if idx is None:
                continue
            out.setdefault(idx, []).extend(self.slices[idx].final_fissures([s]))
        return out


def initial_input(stack: Stack, start: int) -> list[dict]:
    sl = stack.get(start)
    p = sl.prepared
    hint = hint_image(sl.view, p.box, boundary_polylines(sl.view, p.box, p.scale), p.scale)
    if sl.plain:
        hint = hint.resize(sl.main.size, Image.LANCZOS)
    msg = f"Find and trace the fissures in slice {start}. Image 1 is slice {start}, image 2 has the approximate lobe boundaries. You can scroll to other slices with the tools."
    return [{"role": "user", "content": [text(msg), image(sl.main), image(hint)]}]


def run_scroll(case, cfg: RunConfig, log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    reviewer = Reviewer(client, cfg.reviewer_model, cfg.max_reviews) if cfg.use_reviewer else None
    stack = Stack(case, cfg.plane, cfg.use_tracer, reviewer, cfg.plain_images)
    if not stack.lo <= cfg.start <= stack.hi:
        raise ValueError(f"start slice {cfg.start} outside the lung range {stack.lo}-{stack.hi}")
    top = facts(stack.get(cfg.start).view)["top"]
    template = PROMPTS[cfg.use_tracer]
    if cfg.plain_images:
        for old, new in PLAIN_PROMPT_EDITS:
            assert old in template, old
            template = template.replace(old, new)
    if cfg.use_reviewer:
        template = template.replace("\n\nSuggested way to work", REVIEW_PARAGRAPH + "\n\nSuggested way to work", 1)
    instructions = template.format(max_reviews=cfg.max_reviews, start=cfg.start, plane=cfg.plane, top=top, lo=stack.lo, hi=stack.hi, mm_step=stack.mm_step, five_mm=max(round(5 / stack.mm_step), 1))
    kw = dict(model=cfg.model, instructions=instructions, tools=scroll_tools(cfg.use_tracer, cfg.use_reviewer, cfg.plain_images),      # includes submit (it is part of TOOLS)
              reasoning=cfg.reasoning, max_output_tokens=cfg.max_output_tokens)
    finished, transcript, usage, steps = converse(client, kw, initial_input(stack, cfg.start),
                                                  lambda c, k: stack.call(c.name, json.loads(c.arguments), log_dir, k, c.call_id), "submit", cfg.max_steps, single_call=cfg.one_call_per_turn)
    start = cfg.start
    submitted = finished["lines"] if finished else [{"line_id": i, "status": "uncertain", "geometry": "mine", "evidence": "agent did not submit"} for i in stack.owner]
    by_slice = stack.final_by_slice(submitted)
    (log_dir / "transcript.json").write_text(json.dumps({"transcript": transcript, "submitted": submitted, "usage": usage}, indent=1))
    visited = sorted({c["args"].get("slice") for s in transcript for c in s["calls"] if c["args"].get("slice") is not None})
    return {"fissures": by_slice.get(start, []), "extra": {k: v for k, v in by_slice.items() if k != start}, "steps": steps, "usage": usage,
            "visited": visited, "stack": stack, "start": start, "submitted_explicitly": finished is not None}


def run_one(case, name: str, plane: str, idx: int, cfg_base: RunConfig) -> dict:
    d = OUT / name / f"{plane}_{idx}"
    tag = "AGP" if cfg_base.plain_images else ("AGR" if cfg_base.use_reviewer else ("AGS" if cfg_base.use_tracer else "AGN"))          # AGN = no tracer; AGR = no tracer plus the reviewer
    cfg = RunConfig(plane=plane, start=idx, model=cfg_base.model, use_tracer=cfg_base.use_tracer, one_call_per_turn=cfg_base.one_call_per_turn, max_steps=cfg_base.max_steps,
                    use_reviewer=cfg_base.use_reviewer, max_reviews=cfg_base.max_reviews, reviewer_model=cfg_base.reviewer_model, plain_images=cfg_base.plain_images)
    try:
        r = run_scroll(case, cfg, d / ("agent_log_scroll_plain" if cfg.plain_images else "agent_log_scroll_review" if cfg.use_reviewer else ("agent_log_scroll" if cfg.use_tracer else "agent_log_scroll_notracer")))   # a fresh stack per run: lines and views are per agent
    except Exception as e:
        return {"case": name, "plane": plane, "index": idx, "variant": tag, "error": str(e)[:300]}
    sl = r["stack"].get(idx)
    (d / f"{tag}.json").write_text(json.dumps(r["fissures"], indent=2))
    (d / f"{tag}_extra.json").write_text(json.dumps({str(k): v for k, v in r["extra"].items()}, indent=2))
    polys = boundary_polylines(sl.view, sl.prepared.box, sl.prepared.scale)
    panel(sl.prepared, hint_image(sl.view, sl.prepared.box, polys, sl.prepared.scale), draw_model_lines(sl.prepared.images[0], r["fissures"])).save(d / f"{tag}.panel.png")
    return {"case": name, "plane": plane, "index": idx, "variant": tag, "n_fissures": len(r["fissures"]), "steps": r["steps"], "tokens": r["usage"],
            "visited": r["visited"], "extra_slices_with_lines": sorted(r["extra"]), **proxy_scores(sl.view, r["fissures"], sl.prepared)}


def main():
    "scroll_agent <cases.json> [model] [only_case_substring] [--no-tracer] [--max-steps N] [--review] [--plain]"
    max_steps = int(sys.argv[sys.argv.index("--max-steps") + 1]) if "--max-steps" in sys.argv else RunConfig(plane="coronal", start=0).max_steps
    args = [a for a in sys.argv[1:] if a not in ("--no-tracer", "--max-steps", "--review", "--plain") and a != str(max_steps)]
    use_reviewer = "--review" in sys.argv
    plain = "--plain" in sys.argv
    use_tracer = "--no-tracer" not in sys.argv
    cases = json.load(open(args[0]))
    model = args[1] if len(args) > 1 else MODEL
    only = args[2] if len(args) > 2 else ""
    cfg_base = RunConfig(plane="coronal", start=0, model=model, use_tracer=use_tracer, max_steps=max_steps, use_reviewer=use_reviewer, plain_images=plain)      # plane and start are filled in per job
    jobs = []
    for name, (scan, lobe_dir) in cases.items():
        if only not in name:
            continue
        case = load_case(scan, lobe_dir)
        for plane, idx in json.load(open(OUT / name / "picks.json")):
            jobs.append((case, name, plane, idx))
    with ThreadPoolExecutor(5) as ex:
        results = list(ex.map(lambda j: run_one(*j, cfg_base), jobs))
    (OUT / ("results_scroll_plain.json" if plain else "results_scroll_review.json" if use_reviewer else ("results_scroll.json" if use_tracer else "results_scroll_notracer.json"))).write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
