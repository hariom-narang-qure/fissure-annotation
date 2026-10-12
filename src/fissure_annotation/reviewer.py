"""A separate reviewer agent (a fresh Sol call with no access to the drawing agent's reasoning) that checks drawn lines against a rule list.

Called from `overlay(review=true)`. It sees, for every line, a panel (clean crop | high-contrast crop | crop with the line and numbered point
rings) plus an overview, and answers per line and per point which rule is broken and why. It returns words only: coordinates differ between
agents, so it gives no coordinates and no corrected points. The rules live in prompts/fissure_reviewer_rules.txt so they can be edited.
"""
import json
import re
from pathlib import Path

from PIL import Image

from fissure_annotation.agent import FULL, image, text
from fissure_annotation.drawing.compose import tile_row
from fissure_annotation.drawing.lines import LINE_COLOURS
from fissure_annotation.drawing.rings import draw_point_rings
from fissure_annotation.variants import facts

PROMPTS = Path(__file__).parents[2] / "prompts"
RULES = (PROMPTS / "fissure_reviewer_rules.txt").read_text()
PROMPT = (PROMPTS / "fissure_reviewer.txt").read_text()
TILE_LONG_SIDE = 520
MIN_CROP_UNITS, MARGIN_FRACTION, MARGIN_UNITS = 220, 0.15, 50
TEMP_VIEW = "_review"


class Reviewer:
    def __init__(self, client, model: str, max_reviews: int = 2):
        self.client, self.model, self.max_reviews, self.count = client, model, max_reviews, 0

    def review(self, sl, index: int, line_ids, log_dir: Path, step: int) -> list[dict]:
        if self.count >= self.max_reviews:
            return [text(f"review not run: the limit of {self.max_reviews} reviews per run is used up")]
        ids = [i for i in (line_ids or sorted(sl.lines)) if i in sl.lines]
        if not ids:
            return [text("review not run: there are no lines on this slice to review")]
        self.count += 1
        content = [{"type": "input_text", "text": f"Slice {index}. Lines to review: {ids}."}]
        for i in ids:
            content += [{"type": "input_text", "text": f"Panel for line {i} ({sl.lines[i]['name']}, {sl.lines[i]['side']} lung, {len(sl.lines[i]['mine'])} points):"},
                        {"type": "input_image", "image_url": image(line_panel(sl, i))["image_url"]}]
        overview, _ = sl.render(FULL)
        sl.draw_lines(overview, FULL, ids)
        content += [{"type": "input_text", "text": "Overview of the slice with all the lines:"}, {"type": "input_image", "image_url": image(overview)["image_url"]}]
        f = facts(sl.view)
        resp = self.client.responses.create(model=self.model, instructions=PROMPT.format(plane=f["plane"], top=f["top"], rules=RULES),
                                            input=[{"role": "user", "content": content}], reasoning={"effort": "medium"}, max_output_tokens=12000)
        raw = resp.output_text
        (log_dir / f"step{step:02d}_review_s{index}.json").write_text(raw)
        return [text(format_findings(raw, ids))]


def line_panel(sl, lid: int) -> Image.Image:
    "clean | high contrast | line + numbered rings, all on a crop around the line"
    line = sl.lines[lid]
    xs, ys = [p["x"] for p in line["mine"]], [p["y"] for p in line["mine"]]
    pad_x, pad_y = max(MARGIN_FRACTION * (max(xs) - min(xs)), MARGIN_UNITS), max(MARGIN_FRACTION * (max(ys) - min(ys)), MARGIN_UNITS)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_x, half_y = max((max(xs) - min(xs)) / 2 + pad_x, MIN_CROP_UNITS / 2), max((max(ys) - min(ys)) / 2 + pad_y, MIN_CROP_UNITS / 2)
    box = (max(cx - half_x, 0), max(cy - half_y, 0), min(cx + half_x, 1000), min(cy + half_y, 1000))
    tiles = []
    for contrast in ("normal", "high"):
        sl.views[TEMP_VIEW] = {"box": box, "contrast": contrast}
        img, _ = sl.render(TEMP_VIEW)
        tiles.append(img)
    del sl.views[TEMP_VIEW]
    drawn = tiles[0].copy()
    w, h = drawn.size
    pts = [(lx / 1000 * w, ly / 1000 * h) for lx, ly in (sl.main_to_local(p["x"], p["y"], box) for p in line["mine"])]
    draw_point_rings(drawn, pts, join_colour=LINE_COLOURS[0])
    tiles.append(drawn)
    scale = TILE_LONG_SIDE / max(w, h)
    tiles = [t.resize((round(w * scale), round(h * scale)), Image.LANCZOS) for t in tiles]
    out = tile_row(tiles)
    return out


def format_findings(raw: str, ids: list[int]) -> str:
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return "Reviewer reply could not be read as JSON; raw reply:\n" + raw[:1500]
    lines = ["Reviewer findings (an independent reviewer looked at your lines against a rule list; it can be wrong, so verify each finding by zooming before you change anything, and keep a line you can see is right):"]
    for entry in data.get("lines", []):
        probs = entry.get("problems", [])
        if not probs:
            lines.append(f"- line {entry.get('line_id')}: ok")
            continue
        lines.append(f"- line {entry.get('line_id')}: PROBLEMS")
        for p in probs:
            pts = p.get("points") or []
            lines.append(f"    [{p.get('rule')}] " + (f"points {pts}: " if pts else "whole line: ") + str(p.get("why", "")))
    for m in data.get("possibly_missed", []):
        lines.append(f"- possibly missed fissure ({m.get('where')}): {m.get('why')}")
    return "\n".join(lines)
