"""Band masks that tell the agent where each fissure could be, from an automatic lobe segmentation.

Why:
- The agent marks vessels and lobe-like edges as fissures, and misses faint fissures, on a slice seen alone.
- A rough band around each lobe boundary narrows where to look.

How:
1. The lobes come from TotalSegmentator, never from the manual lobes that score the run.
2. For each fissure, the voxels within `band_mm` of the boundary between its lobes form its band (imaging/lobes.py).
   - So a band is about twice `band_mm` across.
   - The work is done in the box around the lungs, since distance transforms of the full volume are slow.
3. Each view is cut into PNGs like the CT slices, on the same pixel grid, each pixel holding its fissure label.
4. The agent sees the band edges as dashed lines on the slices of its first message, one colour per fissure (`OUTLINE_COLOUR_OF_FISSURE`).
   - The prompt names the colours, so the agent knows which fissure a pair of lines is for.
   - Zooms and trace overlays stay clean, since lines drawn there hid the fissure they were meant to point to.

What was tried on 40 LOCCA slices (2026-10-29), before settling on this:
- A separate band mask image after each slice: the agent could not relate it to the slice, and dropped whole fissures.
- The guides on every picture: they hid the fissure in places.
- Guides 10 mm each side of the boundary: worse on precision and recall.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from fissure_annotation.imaging.lobes import LEFT_OBLIQUE, RIGHT_HORIZONTAL, RIGHT_OBLIQUE, fissure_band_labels
from fissure_annotation.imaging.slices import View, write_label_slices

# The guides take a colour per fissure, none of them red, which marks the agent's own trace.
OUTLINE_COLOUR_OF_FISSURE = {
    LEFT_OBLIQUE: (0, 220, 255),  # cyan
    RIGHT_OBLIQUE: (255, 220, 0),  # yellow
    RIGHT_HORIZONTAL: (0, 255, 0),  # green
}
DONE_MARKER = "done"


def band_labels_of(lobes_path: Path, band_mm: float) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(lobes_path))
    lobes = np.asarray(image.dataobj).astype(np.uint8)
    spacing = tuple(float(s) for s in image.header.get_zooms()[:3])
    labels = np.zeros(lobes.shape, dtype=np.uint8)
    box = ndimage.find_objects((lobes > 0).astype(np.uint8))[0]
    labels[box] = fissure_band_labels(lobes[box], spacing, band_mm)
    return labels


def write_hint_slices(lobes_path: Path, scan_path: Path, out_dir: Path, views: list[View], band_mm: float) -> None:
    "a folder per view; a finished folder has a marker file and is not written again"
    pending = [view for view in views if not (out_dir / view / DONE_MARKER).exists()]
    if not pending:
        return
    require_same_grid(lobes_path, scan_path)
    labels = band_labels_of(lobes_path, band_mm)
    for view in pending:
        write_label_slices(labels, out_dir / view, view)
        (out_dir / view / DONE_MARKER).touch()


def require_same_grid(lobes_path: Path, scan_path: Path) -> None:
    "the band PNGs must sit on the pixels of the CT PNGs, which a resampled scan would break"
    lobes, scan = nib.load(lobes_path), nib.load(scan_path)
    if lobes.shape[:3] != scan.shape[:3] or not np.allclose(lobes.affine, scan.affine, atol=1e-3):
        raise ValueError(f"{lobes_path} is not on the grid of {scan_path}")
