"""Turn Molmo's points into stack-agent run folders, so the repo's scoring treats them like the agent's lines.

Steps per slice:
1. Round the points to pixels and drop duplicates.
2. Group them into fissures by single-linkage clustering: points closer than LINK_PX join the same fissure.
   - Molmo's object ids are not used. It gives every point its own id, so they do not say which line a point is on.
3. Order the points of each group by their projection on the group's first principal axis.
   - A group of one point cannot be traced and is dropped from the lines. It still counts for raw point accuracy.
4. Trace and save exactly as StackSession.save_slice does: waypoints, mask from trace_mask on the CT, red overlay on the picture Molmo saw.
5. Write a minimal StackLog per stack, so run_stack_experiment.py --score-only finds the conversation done.

Raw point accuracy: the share of points within 3 mm of a manual lobe border.
Each point is also labelled by what it lies on, in the order of what_traced.py: border, outside the lung, rim, vessel, plain lung.

Usage: molmo_to_runs.py <results.jsonl> <prefix for out dirs and configs>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage

from fissure_annotation.imaging.line_scores import near, reference_fissure_lines
from fissure_annotation.imaging.overlay import red_overlay
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.imaging.slices import slice_view, view_pixel_spacing
from fissure_annotation.imaging.tracing import trace_mask
from fissure_annotation.settings import load_settings
from fissure_annotation.stack_agent import StackFolder, StackLog
from fissure_annotation.stack_experiment import StackExperiment
from fissure_annotation.waypoints import Fissure, SliceWaypoints

EVAL_CONFIG = Path("configs/stack_agent_eval40_step3_ridge_no_guides.yaml")
INPUTS = Path("work/molmo/inputs")
LINK_PX = 50
TOLERANCE_MM = 3.0


def unique_pixels(points: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    "(n, 2) as (x, y); points are (object_id, image_index, x, y)"
    if not points:
        return np.zeros((0, 2), dtype=int)
    xy = np.rint(np.asarray(points)[:, 2:4]).astype(int)
    xy[:, 0] = xy[:, 0].clip(0, shape[1] - 1)
    xy[:, 1] = xy[:, 1].clip(0, shape[0] - 1)
    return np.unique(xy, axis=0)


def group_points(xy: np.ndarray) -> list[np.ndarray]:
    if len(xy) < 2:
        return [xy] if len(xy) else []
    labels = fcluster(linkage(xy.astype(float), method="single"), t=LINK_PX, criterion="distance")
    return [xy[labels == label] for label in np.unique(labels)]


def order_along_main_axis(xy: np.ndarray) -> np.ndarray:
    centred = xy - xy.mean(axis=0)
    _, _, axes = np.linalg.svd(centred.astype(float), full_matrices=False)
    return xy[np.argsort(centred @ axes[0])]


def fissures_of(xy: np.ndarray) -> list[Fissure]:
    groups = [order_along_main_axis(g) for g in group_points(xy) if len(g) >= 2]
    return [Fissure(name=f"molmo_{i}", points=[(int(x), int(y)) for x, y in g]) for i, g in enumerate(groups)]


class Lobes:
    "manual lobes and CT of each scan, cached, cut into slices like the slice PNGs"

    def __init__(self):
        self.cache = {}

    def volumes(self, run_dir: Path):
        if run_dir not in self.cache:
            lobes = nib.as_closest_canonical(nib.load(run_dir / "lobes.nii.gz"))
            scan = nib.as_closest_canonical(nib.load(run_dir / "scan.nii.gz"))
            spacing = tuple(float(s) for s in lobes.header.get_zooms()[:3])
            self.cache[run_dir] = (np.asarray(lobes.dataobj).astype(np.int16), np.asarray(scan.dataobj), spacing)
        return self.cache[run_dir]

    def point_regions(self, stack, z: int) -> list[tuple[str, np.ndarray]]:
        lobes, hu, spacing = self.volumes(stack.run_dir)
        px = view_pixel_spacing(spacing, stack.view)
        lobe, ct = slice_view(lobes, stack.view, z), slice_view(hu, stack.view, z)
        lung = lobe > 0
        rim = lung & ~ndimage.binary_erosion(lung, np.ones((3, 3), bool))
        return [
            ("lobe border", near(reference_fissure_lines(lobe), px, TOLERANCE_MM)),
            ("outside lung", ~ndimage.binary_dilation(lung, iterations=2)),
            ("lung rim", near(rim, px, 3)),
            ("vessel", near(ct > -500, px, 2) & lung),
            ("plain lung", np.ones_like(lung)),
        ]


def label_points(regions, xy: np.ndarray) -> list[str]:
    return [next(name for name, region in regions if region[y, x]) for x, y in xy]


def write_config(prefix: str, variant: str, out_dir: Path, kind: str) -> Path:
    config = yaml.safe_load(EVAL_CONFIG.read_text())
    config.update(out_dir=str(out_dir), repeats=1, pictures={"kind": kind, "display_max": 30} if kind == "ridge" else {"kind": "ct"})
    path = Path(f"configs/{prefix}_{variant}.yaml")
    path.write_text(f"# Molmo points on the 40 eval slices of {EVAL_CONFIG.name}, traced and saved by work/molmo/scripts/molmo_to_runs.py.\n" + yaml.safe_dump(config, sort_keys=False))
    return path


def main(results: Path, prefix: str):
    job = load_settings(StackExperiment, ["--config", str(EVAL_CONFIG)])
    stacks = {s.name: s for s in job.stacks}
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[(row["input"], row["query"])].append(row)
    lobes, summary = Lobes(), {}
    for (kind, query), variant_rows in sorted(by_variant.items()):
        variant = f"{kind}_{query}"
        out_dir = Path(f"work/stack_agent/{prefix}_{variant}")
        labels, per_stack = Counter(), defaultdict(list)
        for row in variant_rows:
            stack = stacks[row["stack"]]
            png = stack.view_dir / "slices" / f"slice_{row['slice']:04d}.png"
            image = read_gray(png)
            picture = read_gray(INPUTS / kind / row["name"])
            folder = StackFolder(root=out_dir / stack.name / "size5" / "repeat0")
            xy = unique_pixels(row["points"], image.shape)
            fissures = fissures_of(xy)
            folder.save_waypoints(png, SliceWaypoints(slice_index=row["slice"], fissures=fissures, notes=f"{len(xy)} molmo points"))
            raw = folder.root / "raw_points" / f"{png.stem}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps({"points_xy": xy.tolist(), "text": row["text"]}))
            if fissures:
                mask = trace_mask(image, [f.points for f in fissures])
                save_png(mask.astype(np.uint8) * 255, folder.mask_path(png))
                save_png(red_overlay(picture, mask), folder.overlay_path(png))
            else:
                folder.mask_path(png).unlink(missing_ok=True)
                folder.overlay_path(png).unlink(missing_ok=True)
            labels.update(label_points(lobes.point_regions(stack, row["slice"]), xy))
            per_stack[stack.name].append(row["slice"])
        for name, slices in per_stack.items():
            folder = StackFolder(root=out_dir / name / "size5" / "repeat0")
            log = StackLog(slice_indices=sorted(slices), summary="molmo points", traces=1, steps=1, input_tokens=0, output_tokens=0, calls=[])
            path = folder.stack_log_path(sorted(slices))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(log.model_dump_json(indent=1))
        config = write_config(prefix, variant, out_dir, kind)
        total = sum(labels.values())
        summary[variant] = {
            "config": str(config), "slices": len(variant_rows), "points": total,
            "points_per_slice": total / len(variant_rows),
            "raw_point_accuracy": labels["lobe border"] / total if total else None,
            "point_labels": {k: v / total for k, v in labels.most_common()} if total else {},
        }
        print(variant, json.dumps(summary[variant]))
    Path(f"work/molmo/{prefix}_point_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
