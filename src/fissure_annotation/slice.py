"""Pull one lung-window 2D slice out of a NIfTI CT volume as an 8-bit PNG.

Orientation is normalised with nibabel's closest-canonical (RAS+) so that axis 0 runs left->right,
axis 1 posterior->anterior, axis 2 inferior->superior. Window is centre -600, width 1500 (a common lung window).
"""
import sys

import nibabel as nib
import numpy as np
from PIL import Image

LUNG_CENTRE, LUNG_WIDTH = -600, 1500


def lung_window(hu: np.ndarray) -> np.ndarray:
    lo = LUNG_CENTRE - LUNG_WIDTH / 2
    return (np.clip((hu - lo) / LUNG_WIDTH, 0, 1) * 255).astype(np.uint8)


def coronal(vol: np.ndarray, frac: float) -> np.ndarray:
    "vol is RAS+; returns rows=superior->inferior, cols=patient right->left (radiological display)"
    y = int(frac * (vol.shape[1] - 1))
    return vol[:, y, :].T[::-1]


def axial(vol: np.ndarray, frac: float) -> np.ndarray:
    z = int(frac * (vol.shape[2] - 1))
    return vol[:, :, z].T[::-1]


def main():
    "slice <nii.gz> <coronal|axial> <frac 0-1> <out.png>"
    path, plane, frac, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    nii = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    print("shape", vol.shape, "spacing", nii.header.get_zooms(), "HU range", vol.min(), vol.max())
    pick = coronal if plane == "coronal" else axial
    Image.fromarray(lung_window(pick(vol, frac))).save(out)


if __name__ == "__main__":
    main()
