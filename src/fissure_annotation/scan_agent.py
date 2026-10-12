"""Per-scan fissure agent: one conversation over a whole CT scan.

The model gets an overview montage of the scan, chooses slices with `open_slice`, and then uses the same
zoom / snap / overlay tools as the slice agent on the slice it has open. It finishes with `submit_scan`: for each of
right oblique, right horizontal, left oblique (and accessory) a presence call plus the traces that show it.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from portkey_ai import Portkey

from fissure_annotation.agent import REASONING, TOOLS, Slice, converse, rgb_data_url, thin
from fissure_annotation.drawing.model_lines import draw_model_lines
from fissure_annotation.drawing.panels import contact_sheet, labelled_thumbnail
from fissure_annotation.experiment import OUT, panel
from fissure_annotation.slicing import cut, load_case
from fissure_annotation.variants import boundary_polylines, hint_image

MODEL = "@openai/gpt-6.1-sol"
PROMPT = (Path(__file__).parents[2] / "prompts" / "fissure_scan_agent.txt").read_text()
MAX_STEPS = 50
OVERVIEW_FRACTIONS = {"coronal": [0.3, 0.45, 0.6, 0.75], "axial": [0.2, 0.4, 0.6, 0.8]}
THUMB = 360
FISSURE_NAMES = {"right_oblique": ("oblique", "right"), "right_horizontal": ("horizontal", "right"), "left_oblique": ("oblique", "left")}
AXIS = {"coronal": 1, "axial": 2}

SCAN_TOOLS = [
    {"type": "function", "name": "open_slice", "description": "Open a coronal or axial slice; it becomes the current slice. Returns the slice and the same slice with approximate lobe boundaries.",
     "parameters": {"type": "object", "properties": {"plane": {"type": "string", "enum": ["coronal", "axial"]}, "index": {"type": "integer"}}, "required": ["plane", "index"]}},
    *[t for t in TOOLS if t["name"] in ("zoom", "snap", "overlay")],
    {"type": "function", "name": "submit_scan", "description": "Finish with the per-fissure findings for the whole scan.",
     "parameters": {"type": "object", "properties": {
         "fissures": {"type": "array", "items": {"type": "object", "properties": {
             "fissure": {"type": "string", "enum": ["right_oblique", "right_horizontal", "left_oblique", "accessory"]},
             "side": {"type": "string", "enum": ["right", "left"], "description": "only needed for accessory"},
             "presence": {"type": "string", "enum": ["present", "absent", "uncertain"]},
             "confidence": {"type": "string", "enum": ["clear", "faint"]},
             "evidence": {"type": "string"},
             "traces": {"type": "array", "items": {"type": "object", "properties": {
                 "plane": {"type": "string", "enum": ["coronal", "axial"]}, "index": {"type": "integer"}, "line_id": {"type": "integer"},
                 "geometry": {"type": "string", "enum": ["mine", "traced"]}}, "required": ["plane", "index", "line_id", "geometry"]}}},
             "required": ["fissure", "presence", "confidence", "evidence", "traces"]}},
         "summary": {"type": "string"}}, "required": ["fissures", "summary"]}},
]


class ScanSession:
    def __init__(self, case):
        self.case = case
        self.slices: dict[tuple[str, int], Slice] = {}
        self.current: tuple[str, int] | None = None
        self.extent = {p: self.lung_extent(a) for p, a in AXIS.items()}

    def lung_extent(self, axis: int) -> tuple[int, int]:
        present = np.nonzero((self.case.lobes > 0).sum(axis=tuple(i for i in range(3) if i != axis)) > 500)[0]
        return int(present[0]), int(present[-1])

    def get(self, plane: str, index: int) -> Slice:
        key = (plane, index)
        if key not in self.slices:
            self.slices[key] = Slice(cut(self.case, plane, index))
        return self.slices[key]

    def overview(self) -> Image.Image:
        tiles = []
        for plane, fracs in OVERVIEW_FRACTIONS.items():
            lo, hi = self.extent[plane]
            for f in fracs:
                idx = int(lo + f * (hi - lo))
                img = self.get(plane, idx).main
                tiles.append(labelled_thumbnail(img, THUMB, f"{plane}:{idx}"))
        return contact_sheet(tiles, columns=4)

    def current_slice(self) -> Slice:
        if self.current is None:
            raise RuntimeError("no slice is open: call open_slice first")
        return self.slices[self.current]

    # ---- tools
    def open_slice(self, plane: str, index: int):
        lo, hi = self.extent[plane]
        if not lo <= index <= hi:
            return [{"type": "input_text", "text": f"index out of range: {plane} slices with lung are {lo}-{hi}"}]
        self.opened.add((plane, index))
        sl = self.get(plane, index)
        self.current = (plane, index)
        p = sl.prepared
        hint = hint_image(sl.view, p.box, boundary_polylines(sl.view, p.box, p.scale), p.scale)
        n = len(sl.lines)
        return [{"type": "input_text", "text": f"opened {plane}:{index} (lung slices {lo}-{hi} in this plane); image 1 = slice, image 2 = approximate lobe boundaries; this slice has {n} traced line(s)"},
                {"type": "input_image", "image_url": rgb_data_url(sl.main)}, {"type": "input_image", "image_url": rgb_data_url(hint)}]

    opened: set = None

    def submit(self, args: dict) -> dict:
        "per-slice final lines from the traces the model listed, plus the per-fissure findings"
        by_slice: dict[tuple[str, int], list] = {}
        for f in args["fissures"]:
            name, side = FISSURE_NAMES.get(f["fissure"], ("accessory", f.get("side", "unknown")))
            for t in f["traces"]:
                key = (t["plane"], t["index"])
                sl = self.slices.get(key)
                if sl is None or t["line_id"] not in sl.lines:
                    continue
                status = "confirmed" if f["confidence"] == "clear" and f["presence"] == "present" else "uncertain"
                line = sl.final_fissures([{"line_id": t["line_id"], "status": status, "geometry": t["geometry"], "evidence": f["evidence"]}])[0]
                line.update(name=name, side=side)
                by_slice.setdefault(key, []).append(line)
        return by_slice


def run_scan(case_name: str, case, model: str, log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    sess = ScanSession(case)
    sess.opened = set()
    ov = sess.overview()
    ov.save(log_dir / "overview.png")
    rng = ", ".join(f"{p} {a}-{b}" for p, (a, b) in sess.extent.items())
    first = [{"role": "user", "content": [
        {"type": "input_text", "text": f"Overview of the scan (thumbnails labelled plane:index; slice indices that contain lung: {rng}). Find the fissures."},
        {"type": "input_image", "image_url": rgb_data_url(ov)}]}]
    kw = dict(model=model, instructions=PROMPT, tools=SCAN_TOOLS, reasoning=REASONING, max_output_tokens=24000)

    def handle(call, step):
        args = json.loads(call.arguments)
        if call.name == "open_slice":
            content = sess.open_slice(**args)
        else:
            sl = sess.current_slice()
            content, img = {"snap": sl.snap_tool, "overlay": sl.overlay_tool, "zoom": sl.zoom_tool}[call.name](**args)
            if img is not None:
                img.save(log_dir / f"step{step:02d}_{call.name}_{call.call_id[-6:]}.png")
        return content

    finished, transcript, usage, steps = converse(client, kw, first, handle, "submit_scan", MAX_STEPS)
    (log_dir / "transcript.json").write_text(json.dumps({"transcript": transcript, "submitted": finished, "usage": usage}, indent=1))
    return {"sess": sess, "submitted": finished, "usage": usage, "steps": steps}


def write_outputs(case_name: str, r: dict) -> dict:
    sess, sub = r["sess"], r["submitted"]
    out = OUT / case_name
    if sub is None:
        return {"case": case_name, "error": "no submit_scan", "steps": r["steps"], "tokens": r["usage"]}
    (out / "scan_summary.json").write_text(json.dumps(sub, indent=1))
    by_slice = sess.submit(sub)
    for (plane, idx), lines in by_slice.items():
        d = out / f"{plane}_{idx}"
        d.mkdir(parents=True, exist_ok=True)
        sl = sess.slices[(plane, idx)]
        (d / "SA.json").write_text(json.dumps(lines, indent=2))
        polys = boundary_polylines(sl.view, sl.prepared.box, sl.prepared.scale)
        panel(sl.prepared, hint_image(sl.view, sl.prepared.box, polys, sl.prepared.scale), draw_model_lines(sl.main, lines)).save(d / "SA.panel.png")
    return {"case": case_name, "steps": r["steps"], "tokens": r["usage"], "opened": sorted(sess.opened),
            "findings": {f["fissure"]: (f["presence"], f["confidence"]) for f in sub["fissures"]}, "slices_with_lines": sorted(by_slice)}


def main():
    "scan_agent <cases.json> [model]"
    cases = json.load(open(sys.argv[1]))
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL

    def one(item):
        name, (scan, lobe_dir) = item
        try:
            return write_outputs(name, run_scan(name, load_case(scan, lobe_dir), model, OUT / name / "scan_agent_log"))
        except Exception as e:
            return {"case": name, "error": str(e)[:300]}

    with ThreadPoolExecutor(5) as ex:
        results = list(ex.map(one, cases.items()))
    (OUT / "results_scan_agent.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
