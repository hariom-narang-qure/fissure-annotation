"""Draw Molmo's points on the slice for review: yellow points, red trace, green manual lobe borders.

The borders are drawn only for the reviewer. Molmo never sees them.
Usage: render_points.py <results.jsonl> <out dir> <input kind> <query id> [run dir prefix] [name filter]
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from fissure_annotation.imaging.line_scores import reference_fissure_lines
from fissure_annotation.imaging.png import read_gray
from fissure_annotation.imaging.slices import slice_view
from fissure_annotation.settings import load_settings
from fissure_annotation.stack_experiment import StackExperiment

EVAL_CONFIG = "configs/stack_agent_eval40_step3_ridge_no_guides.yaml"
SCALE = 2
lobes_cache = {}


def lobe_slice(stack, z):
    if stack.run_dir not in lobes_cache:
        lobes_cache[stack.run_dir] = np.asarray(nib.as_closest_canonical(nib.load(stack.run_dir / "lobes.nii.gz")).dataobj).astype(np.int16)
    return slice_view(lobes_cache[stack.run_dir], stack.view, z)


def panel(image, border, trace, points_xy):
    rgb = np.stack([image] * 3, axis=-1)
    rgb[border] = (0, 200, 0)
    if trace is not None:
        rgb[trace] = (255, 0, 0)
    picture = Image.fromarray(rgb).resize((image.shape[1] * SCALE, image.shape[0] * SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(picture)
    for x, y in points_xy:
        draw.ellipse([x * SCALE - 5, y * SCALE - 5, x * SCALE + 5, y * SCALE + 5], outline=(255, 255, 0), width=2)
    return picture


if __name__ == "__main__":
    results, out, kind, query = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    run_prefix = sys.argv[5] if len(sys.argv) > 5 else ""
    name_filter = sys.argv[6] if len(sys.argv) > 6 else ""
    stacks = {s.name: s for s in load_settings(StackExperiment, ["--config", EVAL_CONFIG]).stacks}
    out.mkdir(parents=True, exist_ok=True)
    for row in map(json.loads, results.read_text().splitlines()):
        if row["input"] != kind or row["query"] != query or name_filter not in row["name"]:
            continue
        stack = stacks[row["stack"]]
        png = stack.view_dir / "slices" / f"slice_{row['slice']:04d}.png"
        image = read_gray(png)
        border = reference_fissure_lines(lobe_slice(stack, row["slice"]))
        mask_path = Path(f"work/stack_agent/{run_prefix}_{kind}_{query}/{stack.name}/size5/repeat0/masks/{png.stem}_mask.png")
        trace = read_gray(mask_path) > 0 if run_prefix and mask_path.exists() else None
        xy = [(p[2], p[3]) for p in row["points"]]
        seen = read_gray(Path("work/molmo/inputs") / kind / row["name"])
        both = Image.new("RGB", (image.shape[1] * SCALE * 2, image.shape[0] * SCALE))
        both.paste(panel(seen, np.zeros_like(border), None, xy), (0, 0))
        both.paste(panel(image, border, trace, xy), (image.shape[1] * SCALE, 0))
        both.save(out / f"{kind}_{query}_{row['name']}")
