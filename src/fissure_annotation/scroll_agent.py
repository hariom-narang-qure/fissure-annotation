"""Slice agent that can scroll: every tool takes the slice number to act on (required), in one plane of one scan.

The model starts on a given slice (named in the prompt) and may look at, zoom into, trace on and compare any other slice
of the same plane to decide whether a line is really a fissure; the lines on the start slice are the deliverable. All
slices of a plane share one crop box (the union of their lung boxes), so a 0-1000 position, and a zoomed view, mean the
same anatomy on every slice. Line ids are global. Every tool can also return the completed slice as submitted, with
or without the lines drawn on it.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from portkey_ai import Portkey

from fissure_annotation.agent import FULL, Slice, converse, image, text
from fissure_annotation.config import RunConfig
from fissure_annotation.reviewer import Reviewer
from fissure_annotation.slicing import cut
from fissure_annotation.tools.toolsets import scroll_tools
from fissure_annotation.variants import boundary_polylines, facts, hint_image, lung_box

PROMPTS = {True: (Path(__file__).parents[2] / "prompts" / "fissure_scroll_agent.txt").read_text(),
           False: (Path(__file__).parents[2] / "prompts" / "fissure_scroll_agent_notracer.txt").read_text()}   # keyed by AgentConfig.use_tracer
AXIS = {"coronal": 1, "axial": 2}
BOX_STRIDE_MM = 8
PLAIN_PROMPT_EDITS = [("; tick marks and labels are drawn inside the edges of zoomed images every 100 units", "; the images have no tick marks or labels, so place points by eye as a fraction of the image width and height (the centre is 500, 500)"),
                      (" clean, with tick marks, in the view", " clean, in the view")]


REVIEW_PARAGRAPH = ("\n\nReview (optional): `overlay(..., review: true)` also asks a separate, independent reviewer to check your lines against a rule list (ends, no interpolation, points on the line, smooth path, no duplicates, right structure, naming, missed fissures) and to tell you which lines, and which numbered points of them, look problematic and why, in words only (no coordinates). "
                    "Use it once when your lines for the slice are drawn, before you submit; you have at most {max_reviews} reviews per run. The reviewer can be wrong: verify each finding by zooming on that spot before you change anything, and keep a line you can see is right.")


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
            tools = sl.tool_methods()
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
