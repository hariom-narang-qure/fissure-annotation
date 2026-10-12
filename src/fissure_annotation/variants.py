"""Build what the model is shown for each experiment variant, and map its answer back to full-slice pixels.

A  facts only: isotropic slice, correct orientation stated.
B  A + crop to the lungs and upscale (a faint fissure gets more pixels).
C  B + text hint: approximate lobe-boundary polylines from the (imperfect) lobe segmentation.
D  B + a second image with those boundaries drawn on the slice.
E  D with the v3 prompt: adds rules for the mistakes reviewers found (apex/pleura lines, extrapolation, duplicates).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from fissure_annotation.slicing import LOBE_NAMES, View, boundary_pair

PROMPT = (Path(__file__).parents[2] / "prompts" / "fissure_v2.txt").read_text()
PROMPT_V3 = (Path(__file__).parents[2] / "prompts" / "fissure_v3.txt").read_text()   # variant E = D + mistake rules
TARGET_LONG_SIDE = 1024
CROP_MARGIN_MM = 6
MIN_COMPONENT_FRACTION = 0.02
PAIR_COLOURS = ["red", "lime", "cyan", "magenta", "orange", "yellow", "deepskyblue"]


@dataclass
class Prepared:
    images: list[Image.Image]
    prompt: str
    box: tuple[int, int, int, int]       # col0,row0,col1,row1 of the crop inside the full slice
    scale: float                         # model-image px per full-slice px


def lung_box(view: View) -> tuple[int, int, int, int]:
    "bounding box of the big lobe components, so stray specks do not stretch the crop"
    lab, n = ndimage.label(view.lobes > 0)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= MIN_COMPONENT_FRACTION * sizes.sum()])
    rows, cols = np.nonzero(keep)
    m = int(CROP_MARGIN_MM / view.mm_per_px)
    h, w = view.image.shape
    return max(cols.min() - m, 0), max(rows.min() - m, 0), min(cols.max() + m, w), min(rows.max() + m, h)


def upscaled(view: View, box) -> tuple[Image.Image, float]:
    c0, r0, c1, r1 = box
    img = Image.fromarray(view.image[r0:r1, c0:c1])
    scale = TARGET_LONG_SIDE / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS), scale


def facts(view: View) -> dict:
    return {"plane": view.plane, "top": "anterior (front of the body)" if view.plane == "axial" else "superior (head end)"}


def boundary_polylines(view: View, box, scale: float, n_points: int = 5) -> list[tuple[str, list[tuple[int, int]]]]:
    "one coarse polyline per touching lobe pair, in 0-1000 coordinates of the model image"
    c0, r0, c1, r1 = box
    out = []
    for a in range(1, 6):
        for b in range(a + 1, 6):
            pair = boundary_pair(view.lobes, a, b)
            lab, n = ndimage.label(pair, structure=np.ones((3, 3)))
            if n == 0:
                continue
            sizes = ndimage.sum(pair, lab, range(1, n + 1))
            rows, cols = np.nonzero(lab == 1 + int(np.argmax(sizes)))
            if len(rows) < 20 / view.mm_per_px:
                continue
            order = np.argsort(cols if np.ptp(cols) >= np.ptp(rows) else rows)
            pts = [(cols[order[i]], rows[order[i]]) for i in np.linspace(0, len(order) - 1, n_points).astype(int)]
            w, h = (c1 - c0), (r1 - r0)
            out.append((f"{LOBE_NAMES[a]} / {LOBE_NAMES[b]}",
                        [(round((x - c0) / w * 1000), round((y - r0) / h * 1000)) for x, y in pts]))
    return out


def hint_text(polys) -> str:
    lines = "\n".join(f"- {name}: " + " -> ".join(f"({x},{y})" for x, y in pts) for name, pts in polys)
    return ("\nHint from an automatic lobe segmentation (it is imperfect: boundaries can be shifted by several mm, "
            "wrong, or missing). A fissure lies where two lobes meet, so these approximate lobe boundaries, in the same "
            "0-1000 coordinates, tell you roughly where to look. Use them only to know where to look; place your "
            "waypoints where you actually see the line in the image, and omit a fissure you cannot see even if a hint "
            "exists:\n" + lines + "\n")


def hint_image(view: View, box, polys, scale: float) -> Image.Image:
    img, _ = upscaled(view, box)
    rgb = img.convert("RGB"); d = ImageDraw.Draw(rgb)
    for k, (name, pts) in enumerate(polys):
        px = [(x / 1000 * rgb.width, y / 1000 * rgb.height) for x, y in pts]
        d.line(px, fill=PAIR_COLOURS[k % len(PAIR_COLOURS)], width=2)
    return rgb


def prepare(view: View, variant: str) -> Prepared:
    f = facts(view)
    if variant == "A":
        box = (0, 0, view.image.shape[1], view.image.shape[0])
        img = Image.fromarray(view.image)
        scale = 1.0
        note, hint, images = "", "", [img]
    else:
        box = lung_box(view)
        img, scale = upscaled(view, box)
        note = "- The image is cropped to the lungs and enlarged.\n"
        hint, images = "", [img]
        if variant in ("C", "D", "E"):
            polys = boundary_polylines(view, box, scale)
            if variant == "C":
                hint = hint_text(polys)
            else:
                images.append(hint_image(view, box, polys, scale))
                hint = ("\nA second image shows the same slice with approximate lobe boundaries from an automatic lobe "
                        "segmentation drawn as coloured lines (imperfect: shifted, wrong or missing boundaries happen). A fissure "
                        "lies where two lobes meet, so use it only to know where to look. Place your waypoints, in the "
                        "coordinates of the FIRST image, where you actually see the line, and omit a fissure you cannot see "
                        "even if a boundary is drawn.\n")
    return Prepared(images, (PROMPT_V3 if variant == "E" else PROMPT).format(crop_note=note, hint_block=hint, **f), box, scale)


def to_slice_pixels(wp: dict, prepared: Prepared) -> tuple[float, float]:
    "model 0-1000 coords -> (col, row) in the full isotropic slice"
    c0, r0, c1, r1 = prepared.box
    return c0 + wp["x"] / 1000 * (c1 - c0), r0 + wp["y"] / 1000 * (r1 - r0)
