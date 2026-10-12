"""Tool-using fissure agent: one continuous conversation (OpenAI Responses API) over a single CT slice.

Tools: zoom (magnified crop with coordinate ticks), snap (ridge tracer through waypoints, returns fit stats),
overlay (clean crop beside crop with lines), submit. The model proposes lines by calling snap, inspects them with
overlay/zoom, redoes or drops them, and calls submit. All coordinates the model sees are 0-1000 in the cropped
first image; geometry is kept in full-slice pixels.

Chat completions refuses function tools while reasoning is on for gpt-6.1-sol, so this uses /v1/responses with
previous_response_id (the server keeps the conversation, including the model's reasoning, between turns).
"""
import base64
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from portkey_ai import Portkey

from fissure_annotation.client import png_data_url
from fissure_annotation.drawing.ticks import draw_ticks
from fissure_annotation.experiment import OUT, panel, proxy_scores
from fissure_annotation.slicing import LOBE_NAMES, cut, load_case, lung_window
from fissure_annotation.tracing import lung_mask, ridge_map, snap
from fissure_annotation.variants import Prepared, boundary_pair, boundary_polylines, facts, hint_image, prepare, to_slice_pixels, upscaled

MODEL = "@openai/gpt-6.1-sol"
PROMPT = (Path(__file__).parents[2] / "prompts" / "fissure_agent.txt").read_text()
MAX_STEPS = 12
REASONING = {"effort": "medium"}
COLOURS = [(255, 40, 40), (255, 220, 0), (0, 255, 120), (0, 200, 255), (255, 0, 255), (255, 140, 0)]
ZOOM_LONG_SIDE = 900

VIEW_PROPS = {
    "view_id": {"type": "string", "description": "the view whose image you are looking at: 'full' (the first image) or an id returned by zoom. Coordinates you give and receive are in this view's own 0-1000 frame."}}
CONTRASTS = {"normal": (-600, 1500), "high": (-650, 700)}      # (centre, width) HU; 'high' stretches the -1000..-300 range where fissures live
PAIR_COLOURS = [(255, 60, 60), (60, 255, 60), (60, 200, 255), (255, 0, 255), (255, 160, 0), (255, 255, 0), (0, 255, 200), (180, 120, 255), (255, 120, 180), (160, 255, 60)]
FULL = "full"

TOOLS = [
    {"type": "function", "name": "zoom",
     "description": "Look closely: a magnified crop of the clean slice as a new view. Coordinates x0,y0,x1,y1 are in the frame of `view_id` (default: the full image), so you can zoom into a zoom. Returns the new view's id; from then on, give and receive coordinates in the new view's own 0-1000 frame (ticks are drawn inside the image edges, in that frame).",
     "parameters": {"type": "object", "properties": {**{k: {"type": "integer", "minimum": 0, "maximum": 1000} for k in ("x0", "y0", "x1", "y1")}, **VIEW_PROPS,
                    "contrast": {"type": "string", "enum": ["normal", "high"], "description": "high = narrower window that makes a faint fissure stand out against the lung"},
                    "lobes": {"type": "boolean", "description": "also draw approximate lobe boundaries from the automatic segmentation (imperfect), with a colour legend"}},
                    "required": ["x0", "y0", "x1", "y1"]}},
    {"type": "function", "name": "snap",
     "description": "Trace a fissure. Give 2-6 waypoints along it (ordered from one end to the other) in the frame of `view_id`. A ridge tracer moves each waypoint across the line to the nearest ridge and joins them by a minimal-cost path. The line is stored (numbered) in full-image coordinates, and you get back the traced points in the same view frame, fit statistics, and the view image with the line drawn (thick coloured = traced, thin white = your own waypoints). Pass line_id to redo an existing line.",
     "parameters": {"type": "object", "properties": {
         "waypoints": {"type": "array", "minItems": 2, "items": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}},
         **VIEW_PROPS,
         "line_id": {"type": "integer", "description": "existing line to replace; omit for a new line"},
         "name": {"type": "string", "enum": ["oblique", "horizontal", "accessory", "unknown"]},
         "side": {"type": "string", "enum": ["right", "left"]}},
         "required": ["waypoints", "name", "side"]}},
    {"type": "function", "name": "overlay",
     "description": "Compare: the clean view next to the same view with lines drawn and numbered (all lines if line_ids omitted; lines outside the view are simply not visible). Use a zoomed view_id to check a line closely.",
     "parameters": {"type": "object", "properties": {**VIEW_PROPS, "line_ids": {"type": "array", "items": {"type": "integer"}},
                    "lobes": {"type": "boolean", "description": "also draw approximate lobe boundaries, with a colour legend"}}}},
    {"type": "function", "name": "submit", "description": "Finish. Lines not listed are dropped.",
     "parameters": {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "object", "properties": {
         "line_id": {"type": "integer"}, "status": {"type": "string", "enum": ["confirmed", "uncertain", "dropped"]},
         "geometry": {"type": "string", "enum": ["mine", "traced"], "description": "which geometry to keep: your own waypoints, or the traced path"},
         "evidence": {"type": "string"}}, "required": ["line_id", "status", "geometry", "evidence"]}}}, "required": ["lines"]}},
]


def thin(path: np.ndarray, n: int) -> np.ndarray:
    "about n evenly spaced vertices of a per-pixel path, keeping both ends"
    idx = np.unique(np.linspace(0, len(path) - 1, n).round().astype(int))
    return path[idx]


def rgb_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def text(msg: str) -> dict:
    return {"type": "input_text", "text": msg}


def image(img: Image.Image) -> dict:
    return {"type": "input_image", "image_url": rgb_data_url(img)}


class Slice:
    """Everything the tools need about one slice, plus the lines drawn so far.

    Frames: the cropped first image is the 'main' frame (x, y in 0-1000). A view is a box inside it; coordinates the
    model gives and receives are in the 0-1000 frame of the view it names. Lines are stored in the main frame.
    """

    def __init__(self, view, box=None, use_tracer=True, plain=False):
        "box: a crop box shared by every slice of a stack, so one frame means the same anatomy on all of them"
        self.view = view
        self.prepared = prepare(view, "E") if box is None else fixed_prepared(view, box)
        self.next_id = None            # a stack sets this so line ids are unique across slices
        self.use_tracer = use_tracer
        self.score = ridge_map(view.image, lung_mask(view.lobes)) if use_tracer else None
        self.lines: dict[int, dict] = {}
        self.plain = plain
        c0, r0, c1, r1 = self.prepared.box
        self.native_long_side = max(c1 - c0, r1 - r0)      # the crop's own size in slice pixels
        self.main = (Image.fromarray(view.image[r0:r1, c0:c1]).convert("RGB") if plain else self.prepared.images[0].convert("RGB"))
        self.views: dict[str, dict] = {FULL: {"box": (0.0, 0.0, 1000.0, 1000.0), "contrast": "normal"}}

    # ---- frames
    def to_model(self, col: float, row: float) -> tuple[int, int]:
        "slice pixel -> main frame"
        c0, r0, c1, r1 = self.prepared.box
        return round((col - c0) / (c1 - c0) * 1000), round((row - r0) / (r1 - r0) * 1000)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        "main frame -> slice pixel (col, row)"
        return to_slice_pixels({"x": x, "y": y}, self.prepared)

    def box(self, view_id: str) -> tuple[float, float, float, float]:
        if view_id not in self.views:
            raise KeyError(f"unknown view_id {view_id!r}; known: {sorted(self.views)}")
        return self.views[view_id]["box"]

    @staticmethod
    def local_to_main(x: float, y: float, box) -> tuple[float, float]:
        x0, y0, x1, y1 = box
        return x0 + x / 1000 * (x1 - x0), y0 + y / 1000 * (y1 - y0)

    @staticmethod
    def main_to_local(x: float, y: float, box) -> tuple[float, float]:
        x0, y0, x1, y1 = box
        return (x - x0) / (x1 - x0) * 1000, (y - y0) / (y1 - y0) * 1000

    # ---- rendering
    def render(self, view_id: str, lobes: bool = False) -> tuple[Image.Image, list[str]]:
        "the view as an image whose frame is exactly 0-1000, ticks drawn inside the edges; second value is the lobe legend"
        v = self.views[view_id]
        (ca, ra), (cb, rb) = self.to_px(v["box"][0], v["box"][1]), self.to_px(v["box"][2], v["box"][3])
        ca, ra, cb, rb = int(max(ca, 0)), int(max(ra, 0)), int(min(cb, self.view.image.shape[1] - 1)) + 1, int(min(rb, self.view.image.shape[0] - 1)) + 1
        arr = lung_window(self.view.hu[ra:rb, ca:cb], *CONTRASTS[v["contrast"]])
        crop = Image.fromarray(arr)
        scale = (self.native_long_side if self.plain else ZOOM_LONG_SIDE) / max(crop.size)
        img = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS).convert("RGB")
        legend = []
        if lobes:
            legend = self.draw_lobes(img, self.view.lobes[ra:rb, ca:cb])
        if not self.plain:
            draw_ticks(img)
        return img, legend

    def draw_lobes(self, img: Image.Image, sub: np.ndarray) -> list[str]:
        legend = []
        px = img.load()
        for k, (a, b) in enumerate([(a, b) for a in range(1, 6) for b in range(a + 1, 6)]):
            m = boundary_pair(sub, a, b)
            if m.sum() < 6:
                continue
            big = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) > 0
            for y, x in zip(*np.nonzero(big)):      # half transparent, so the fissure underneath stays visible
                px[int(x), int(y)] = tuple((o + c) // 2 for o, c in zip(px[int(x), int(y)], PAIR_COLOURS[k]))
            legend.append(f"{COLOUR_NAMES[k]} = {LOBE_NAMES[a]} / {LOBE_NAMES[b]}")
        return legend

    def draw_lines(self, img: Image.Image, view_id: str, ids: list[int]) -> None:
        box = self.box(view_id)
        d = ImageDraw.Draw(img)
        w, h = img.size
        to_img = lambda x, y: (lambda lx, ly: (lx / 1000 * w, ly / 1000 * h))(*self.main_to_local(x, y, box))
        for k, i in enumerate(ids):
            col = COLOURS[k % len(COLOURS)]
            line = self.lines[i]
            own = [to_img(p["x"], p["y"]) for p in line["mine"]]
            if line["path"] is None:             # no tracer: the model's own line, with its points marked
                d.line(own, fill=col, width=2)
                for x, y in own:
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=col)
                xy = own
            else:
                d.line(own, fill=(255, 255, 255), width=1)
                xy = [to_img(*self.to_model(c, r)) for r, c in line["path"]]
                d.line(xy, fill=col, width=2)
            d.text((xy[0][0] + 5, xy[0][1] - 14), f"{i} {line['side'][0].upper()}", fill=col)

    # ---- tools: each returns (content, image or None)
    def zoom_tool(self, x0, y0, x1, y1, view_id=FULL, contrast="normal", lobes=False):
        base = self.box(view_id)
        x0, x1 = sorted((max(x0, 0), min(x1, 1000)))
        y0, y1 = sorted((max(y0, 0), min(y1, 1000)))
        if x1 - x0 < 30 or y1 - y0 < 30:
            return [text("region too small: need at least 30 units on each side")], None
        (mx0, my0), (mx1, my1) = self.local_to_main(x0, y0, base), self.local_to_main(x1, y1, base)
        vid = f"v{len(self.views)}"
        self.views[vid] = {"box": (mx0, my0, mx1, my1), "contrast": contrast}
        img, legend = self.render(vid, lobes)
        (ca, ra), (cb, rb) = self.to_px(mx0, my0), self.to_px(mx1, my1)
        mm = f"about {abs(cb - ca) * self.view.mm_per_px:.0f} x {abs(rb - ra) * self.view.mm_per_px:.0f} mm"
        return [text(f"view_id={vid}: zoom of {view_id} x {x0}-{x1}, y {y0}-{y1} ({mm}), contrast {contrast}. Coordinates in this view are its own 0-1000 frame ({self.tick_note()})."
                     + ("; lobe boundaries: " + "; ".join(legend) if lobes else "")), image(img)], img

    def line_id_for(self, line_id) -> tuple[int, str]:
        "reuse an existing line's id to redo it, otherwise a new id; the note says when a requested id was not usable"
        lid = line_id if line_id in self.lines else (self.next_id() if self.next_id else max(self.lines, default=0) + 1)
        return lid, (f"line_id {line_id} is not a line on this slice, created line {lid}. " if line_id not in (None, lid) else "")

    def check_points_tool(self, waypoints, view_id=FULL, connect=False):
        """preview only: where do these waypoints land on the view? Nothing is stored or traced.
        Points are in the view's own frame, so they are drawn as given: numbered rings (the line should pass through each ring's centre)."""
        self.box(view_id)
        if not waypoints:
            return [text("no waypoints given")], None
        clean, _ = self.render(view_id)
        marked = clean.copy()
        d = ImageDraw.Draw(marked)
        w, h = marked.size
        pts = [(p["x"] / 1000 * w, p["y"] / 1000 * h) for p in waypoints]
        if connect:
            d.line(pts, fill=(255, 255, 0), width=1)
        for k, (x, y) in enumerate(pts, start=1):
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 0, 255), width=2)
            d.text((x + 8, y - 15), str(k), fill=(255, 0, 255))
        both = Image.new("RGB", (w * 2 + 8, h), "black")
        both.paste(clean, (0, 0))
        both.paste(marked, (w + 8, 0))
        outside = [k for k, p in enumerate(waypoints, start=1) if not (0 <= p["x"] <= 1000 and 0 <= p["y"] <= 1000)]
        note = f"view {view_id}: left = clean, right = your {len(pts)} points as numbered rings (nothing stored)" + (f"; points outside this view: {outside}" if outside else "")
        return [text(note), image(both)], both

    def mark_tool(self, waypoints, name, side, view_id=FULL, line_id=None):
        "no-tracer workflow: the line is the model's waypoints joined by straight segments; the picture shows the line and its points"
        box = self.box(view_id)
        main_pts = [self.local_to_main(w["x"], w["y"], box) for w in waypoints]
        lid, note = self.line_id_for(line_id)
        self.lines[lid] = {"name": name, "side": side, "path": None, "stats": {}, "mine": [{"x": round(x, 1), "y": round(y, 1)} for x, y in main_pts]}
        img, _ = self.render(view_id)
        self.draw_lines(img, view_id, [lid])
        local = [self.main_to_local(x, y, box) for x, y in main_pts]
        return [text(note + json.dumps({"line_id": lid, "view_id": view_id, "points": [{"x": round(x), "y": round(y)} for x, y in local]})), image(img)], img

    def snap_tool(self, waypoints, name, side, view_id=FULL, line_id=None):
        box = self.box(view_id)
        main_pts = [self.local_to_main(w["x"], w["y"], box) for w in waypoints]
        rc = [(self.to_px(x, y)[1], self.to_px(x, y)[0]) for x, y in main_pts]
        s = snap(self.score, rc, self.view.mm_per_px)
        lid, note = self.line_id_for(line_id)
        self.lines[lid] = {"name": name, "side": side, "path": s.path, "stats": s.stats,
                           "mine": [{"x": round(x, 1), "y": round(y, 1)} for x, y in main_pts]}
        local = [self.main_to_local(*self.to_model(c, r), box) for r, c in thin(s.path, 8)]
        outside = any(not (0 <= x <= 1000 and 0 <= y <= 1000) for x, y in local)
        img, _ = self.render(view_id)
        self.draw_lines(img, view_id, [lid])
        return [text(note + json.dumps({"line_id": lid, "view_id": view_id, "traced_points": [{"x": round(x), "y": round(y)} for x, y in local],
                                         "part_of_line_outside_this_view": outside, "fit": s.stats})), image(img)], img

    def overlay_tool(self, view_id=FULL, line_ids=None, lobes=False):
        self.box(view_id)
        ids = [i for i in (line_ids or sorted(self.lines)) if i in self.lines]
        clean, legend = self.render(view_id, lobes)
        drawn = clean.copy()
        self.draw_lines(drawn, view_id, ids)
        both = Image.new("RGB", (clean.width * 2 + 8, clean.height), "black")
        both.paste(clean, (0, 0))
        both.paste(drawn, (clean.width + 8, 0))
        legend = ("thin white = your own waypoints, thick coloured = traced path" if self.use_tracer else "your lines, with circles at your points")
        return [text(f"view {view_id}: left = clean, right = lines {ids} drawn ({legend})"
                     + ("; lobe boundaries: " + "; ".join(legend) if lobes else "")), image(both)], both

    def tick_note(self) -> str:
        return "no tick marks: place points by eye as a fraction of the image width and height" if self.plain else "ticks drawn inside the edges"

    def final_fissures(self, submitted):
        out = []
        for s in submitted:
            line = self.lines.get(s["line_id"])
            if line is None or s["status"] == "dropped":
                continue
            geometry = s.get("geometry", "mine") if line["path"] is not None else "mine"
            waypoints = ([{"x": round(p["x"]), "y": round(p["y"])} for p in line["mine"]] if geometry == "mine"
                         else [dict(zip("xy", self.to_model(c, r))) for r, c in thin(line["path"], 12)])
            out.append({"name": line["name"], "side": line["side"], "confidence": {"confirmed": "clear", "uncertain": "faint"}[s["status"]],
                        "waypoints": waypoints, "evidence": s["evidence"], "geometry": geometry, "fit": line["stats"]})
        return out


def fixed_prepared(view, box) -> Prepared:
    img, scale = upscaled(view, box)
    return Prepared([img], "", box, scale)


COLOUR_NAMES = ["red", "green", "light blue", "magenta", "orange", "yellow", "turquoise", "violet", "pink", "lime"]


def initial_input(sl: Slice) -> list[dict]:
    p = sl.prepared
    polys = boundary_polylines(sl.view, p.box, p.scale)
    images = [sl.main, hint_image(sl.view, p.box, polys, p.scale)]
    return [{"role": "user", "content": [{"type": "input_text", "text": "Find and trace the fissures in this slice. Image 1 is the slice, image 2 has the approximate lobe boundaries."}]
             + [{"type": "input_image", "image_url": rgb_data_url(i)} for i in images]}]


def run_tool(sl: Slice, call, log_dir: Path, step: int):
    args = json.loads(call.arguments)
    tools = {"snap": sl.snap_tool, "overlay": sl.overlay_tool, "zoom": sl.zoom_tool}
    if call.name not in tools:
        return [text(f"unknown tool {call.name}")], None
    content, img = tools[call.name](**args)
    if img is not None:
        img.save(log_dir / f"step{step:02d}_{call.name}_{call.call_id[-6:]}.png")
    return content, None


def create(client, kw: dict, single_call: bool, **extra):
    "responses.create, asking the API for one tool call per turn when requested (retrying without the parameter if it is rejected)"
    if single_call:
        try:
            return client.responses.create(parallel_tool_calls=False, **kw, **extra)
        except Exception as e:
            if "parallel_tool_calls" not in str(e):
                raise
    return client.responses.create(**kw, **extra)


def converse(client, kw: dict, first_input, handle, finish: str, max_steps: int, single_call: bool = False):
    """Drive one tool-calling conversation until the model calls `finish` or max_steps is hit.
    handle(call, step) -> tool output content. With single_call, only the first tool call of a turn is run: the model is told the
    others were skipped, so every action sees the result of the one before it. Returns (finish_args or None, transcript, usage, steps)."""
    transcript, usage = [], {"input": 0, "output": 0, "reasoning": 0}
    resp = create(client, kw, single_call, input=first_input)
    finished, steps = None, 0
    while steps < max_steps:
        for k in ("input", "output"):
            usage[k] += getattr(resp.usage, f"{k}_tokens", 0)
        usage["reasoning"] += getattr(getattr(resp.usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        calls = [o for o in resp.output if o.type == "function_call"]
        skipped = calls[1:] if single_call else []
        run = calls[:1] if single_call else calls
        transcript.append({"step": steps, "calls": [{"name": c.name, "args": json.loads(c.arguments)} for c in run], "skipped": [c.name for c in skipped]})
        if not calls:
            break
        outputs = [{"type": "function_call_output", "call_id": c.call_id,
                    "output": "skipped: only one tool call is run per turn; call it again in a later turn if you still need it"} for c in skipped]
        for c in run:
            if c.name == finish:
                finished = json.loads(c.arguments)
                outputs.append({"type": "function_call_output", "call_id": c.call_id, "output": "submitted"})
                continue
            try:
                content = handle(c, steps)
            except Exception as e:
                content = [{"type": "input_text", "text": f"tool error: {e}"}]
            transcript[-1].setdefault("results", []).append([x["text"] for x in content if x["type"] == "input_text"])
            outputs.append({"type": "function_call_output", "call_id": c.call_id, "output": content})
        if finished is not None:
            break
        steps += 1
        resp = create(client, kw, single_call, previous_response_id=resp.id, input=outputs)
    return finished, transcript, usage, steps


def run_agent(view, model: str, log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    sl = Slice(view)
    kw = dict(model=model, instructions=PROMPT.format(plane=facts(view)["plane"], top=facts(view)["top"]), tools=TOOLS,
              reasoning=REASONING, max_output_tokens=24000)
    finished, transcript, usage, steps = converse(client, kw, initial_input(sl), lambda c, k: run_tool(sl, c, log_dir, k)[0], "submit", MAX_STEPS)
    submitted = finished["lines"] if finished else [{"line_id": i, "status": "uncertain", "geometry": "mine", "evidence": "agent did not submit"} for i in sl.lines]
    fissures = sl.final_fissures(submitted)
    (log_dir / "transcript.json").write_text(json.dumps({"transcript": transcript, "submitted": submitted, "usage": usage}, indent=1))
    return {"fissures": fissures, "steps": steps, "usage": usage, "submitted_explicitly": finished is not None, "slice": sl}


def run_one(case: str, plane: str, idx: int, view, model: str) -> dict:
    d = OUT / case / f"{plane}_{idx}"
    try:
        r = run_agent(view, model, d / "agent_log")
    except Exception as e:
        return {"case": case, "plane": plane, "index": idx, "variant": "AG", "error": str(e)[:300]}
    sl = r["slice"]
    (d / "AG.json").write_text(json.dumps(r["fissures"], indent=2))
    from fissure_annotation.experiment import draw_model_lines
    overlay = draw_model_lines(sl.main, r["fissures"])
    polys = boundary_polylines(view, sl.prepared.box, sl.prepared.scale)
    panel(sl.prepared, hint_image(view, sl.prepared.box, polys, sl.prepared.scale), overlay).save(d / "AG.panel.png")
    return {"case": case, "plane": plane, "index": idx, "variant": "AG", "n_fissures": len(r["fissures"]), "steps": r["steps"],
            "tokens": r["usage"], **proxy_scores(view, r["fissures"], sl.prepared)}


def main():
    "agent <cases.json> [model] [only_case_substring]"
    cases = json.load(open(sys.argv[1]))
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    only = sys.argv[3] if len(sys.argv) > 3 else ""
    jobs = []
    for name, (scan, lobe_dir) in cases.items():
        if only not in name:
            continue
        case = load_case(scan, lobe_dir)
        for plane, idx in json.load(open(OUT / name / "picks.json")):
            jobs.append((name, plane, idx, cut(case, plane, idx)))
    with ThreadPoolExecutor(5) as ex:
        results = list(ex.map(lambda j: run_one(*j[:3], j[3], model), jobs))
    (OUT / f"results_AG_{model.split('/')[-1]}.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
