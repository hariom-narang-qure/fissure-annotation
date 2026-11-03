"""Classify each traced pixel by what it lies on: lobe border (fissure), lung rim, vessel, outside the lung, or plain lung."""
import sys, collections
import nibabel as nib, numpy as np
from scipy import ndimage
from fissure_annotation.imaging.line_scores import near, reference_fissure_lines
from fissure_annotation.imaging.png import read_gray
from fissure_annotation.imaging.slices import slice_view, view_pixel_spacing
from fissure_annotation.settings import load_settings
from fissure_annotation.slice_scores import traced_mask
from fissure_annotation.stack_experiment import StackExperiment

cache = {}
def volumes(run_dir):
    if run_dir not in cache:
        lobes = nib.as_closest_canonical(nib.load(run_dir / "lobes.nii.gz"))
        scan = nib.as_closest_canonical(nib.load(run_dir / "scan.nii.gz"))
        cache[run_dir] = (np.asarray(lobes.dataobj).astype(np.int16), np.asarray(scan.dataobj), tuple(float(s) for s in lobes.header.get_zooms()[:3]))
    return cache[run_dir]

for cfg in sys.argv[1:]:
    job = load_settings(StackExperiment, ["--config", f"configs/{cfg}.yaml"])
    counts, per_stack = collections.Counter(), collections.defaultdict(collections.Counter)
    for c in job.conversations():
        lobes, hu, spacing = volumes(c.stack.run_dir)
        px = view_pixel_spacing(spacing, c.stack.view)
        for z in job.slice_indices(c):
            lobe, ct = slice_view(lobes, c.stack.view, z), slice_view(hu, c.stack.view, z)
            traced = traced_mask(job.folder(c), c.stack.view_dir / "slices", z, lobe.shape)
            lung = lobe > 0
            rim = lung & ~ndimage.binary_erosion(lung, np.ones((3, 3), bool))
            classes = [
                ("lobe border (fissure)", near(reference_fissure_lines(lobe), px, 3)),
                ("outside the lung", ~ndimage.binary_dilation(lung, iterations=2)),
                ("lung rim / pleura", near(rim, px, 3)),
                ("vessel (HU > -500 within 2 mm)", near(ct > -500, px, 2) & lung),
                ("plain lung", np.ones_like(lung)),
            ]
            left = traced.copy()
            for name, region in classes:
                hit = left & region; counts[name] += int(hit.sum()); per_stack[c.stack.name][name] += int(hit.sum()); left &= ~region
    total = sum(counts.values())
    print(f"{cfg}: {total} traced px")
    for name, n in counts.most_common(): print(f"   {n/total:5.1%}  {name}")
    for s, cnt in per_stack.items():
        t = sum(cnt.values())
        if t: print("     ", s, ", ".join(f"{k.split(' (')[0]} {v/t:.0%}" for k, v in cnt.most_common(3)))
