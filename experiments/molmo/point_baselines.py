"""Context for raw point accuracy: the chance level, and the same measure on the agent's own waypoints.

- Chance: the share of lung pixels within 3 mm of a manual lobe border, pooled over the 40 slices.
- Agent: the share of the waypoints saved by a stack agent run that lie within 3 mm of a border.
Usage: point_baselines.py <agent config name> ...
"""
import sys

import nibabel as nib
import numpy as np

from fissure_annotation.imaging.line_scores import near, reference_fissure_lines
from fissure_annotation.imaging.slices import slice_view, view_pixel_spacing
from fissure_annotation.settings import load_settings
from fissure_annotation.stack_experiment import StackExperiment

cache = {}


def border_zone(stack, z):
    if stack.run_dir not in cache:
        im = nib.as_closest_canonical(nib.load(stack.run_dir / "lobes.nii.gz"))
        cache[stack.run_dir] = (np.asarray(im.dataobj).astype(np.int16), tuple(float(s) for s in im.header.get_zooms()[:3]))
    lobes, spacing = cache[stack.run_dir]
    lobe = slice_view(lobes, stack.view, z)
    return lobe > 0, near(reference_fissure_lines(lobe), view_pixel_spacing(spacing, stack.view), 3.0)


for config in sys.argv[1:]:
    job = load_settings(StackExperiment, ["--config", f"configs/{config}.yaml"])
    lung_px = zone_px = hits = points = 0
    for c in job.conversations():
        for z in job.slice_indices(c):
            lung, zone = border_zone(c.stack, z)
            if c.repeat == 0:
                lung_px += int(lung.sum()); zone_px += int((zone & lung).sum())
            png = c.stack.view_dir / "slices" / f"slice_{z:04d}.png"
            if job.folder(c).points_path(png).exists():
                for fissure in job.folder(c).load_waypoints(png).fissures:
                    for x, y in fissure.points:
                        points += 1; hits += int(zone[y, x])
    print(f"{config}: chance {zone_px / lung_px:.1%} of lung pixels; agent waypoints {hits}/{points} = {hits / points:.1%} within 3 mm")
