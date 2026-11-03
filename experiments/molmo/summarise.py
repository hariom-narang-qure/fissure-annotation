"""Pixel-pooled precision and recall (score_run) and fissures found, per config and repeat, at 3 mm.

Fissures found, as in the stack_eval fissure_detection_step3.py:
- A fissure is present on a slice when its manual border has at least MIN_REFERENCE_PX pixels.
- It is found when at least MIN_FOUND_PX of those border pixels lie within 3 mm of a trace.
Existing runs are only read; nothing is written into their folders.
Usage: summarise.py <config name> ...
"""
import sys

import nibabel as nib
import numpy as np

from fissure_annotation.imaging.line_scores import near, touching
from fissure_annotation.imaging.lobes import FISSURE_PAIRS_BY_LABEL
from fissure_annotation.imaging.slices import slice_view, view_pixel_spacing
from fissure_annotation.settings import load_settings
from fissure_annotation.slice_scores import score_run, traced_mask
from fissure_annotation.stack_experiment import StackExperiment

MIN_REFERENCE_PX, MIN_FOUND_PX, TOL = 20, 10, 3.0
lobes_cache = {}


def lobes_of(run_dir):
    if run_dir not in lobes_cache:
        im = nib.as_closest_canonical(nib.load(run_dir / "lobes.nii.gz"))
        lobes_cache[run_dir] = (np.asarray(im.dataobj).astype(np.int16), tuple(float(s) for s in im.header.get_zooms()[:3]))
    return lobes_cache[run_dir]


def fissure_line(lobe_slice, label):
    out = np.zeros(lobe_slice.shape, bool)
    for a, b in FISSURE_PAIRS_BY_LABEL[label]:
        out |= touching(lobe_slice == a, lobe_slice == b) | touching(lobe_slice == b, lobe_slice == a)
    return out


def found_flags(stack, run, z):
    lobes, spacing = lobes_of(stack.run_dir)
    lobe_slice = slice_view(lobes, stack.view, z)
    near_trace = near(traced_mask(run, stack.view_dir / "slices", z, lobe_slice.shape), view_pixel_spacing(spacing, stack.view), TOL)
    for label in FISSURE_PAIRS_BY_LABEL:
        ref = fissure_line(lobe_slice, label)
        if ref.sum() >= MIN_REFERENCE_PX:
            yield int((ref & near_trace).sum()) >= MIN_FOUND_PX


def summarise(config: str) -> list[str]:
    job = load_settings(StackExperiment, ["--config", f"configs/{config}.yaml"])
    lines = []
    for repeat in range(job.repeats):
        scores, found = [], []
        for c in job.conversations():
            if c.repeat != repeat or not job.is_done(c):
                continue
            indices = job.slice_indices(c)
            scores += score_run(c.stack.run_dir / "lobes.nii.gz", c.stack.view_dir / "slices", job.folder(c), c.stack.view, indices, TOL)
            for z in indices:
                found += list(found_flags(c.stack, job.folder(c), z))
        traced = sum(s.traced_px for s in scores)
        reference = sum(s.reference_px for s in scores)
        precision = sum(s.traced_near_reference_px for s in scores) / traced if traced else float("nan")
        recall = sum(s.reference_near_traced_px for s in scores) / reference
        lines.append(f"{config} r{repeat}: slices {len(scores)} traced_px {traced} precision {precision:.1%} recall {recall:.1%} found {sum(found)}/{len(found)} ({np.mean(found):.0%})")
    return lines


if __name__ == "__main__":
    for config in sys.argv[1:]:
        print("\n".join(summarise(config)), flush=True)
