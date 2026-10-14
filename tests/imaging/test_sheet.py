import numpy as np
from scipy import ndimage

from fissure_annotation.imaging.sheet import oriented_sheet

PLANE_X = 20
SIZE = 40
INTERIOR = slice(
    12, 28
)  # away from the edges of the plane, where the blur has fewer neighbours


def ragged_plane(seed=0, jitter=1, missing=0.2):
    "a plane at x=20 that jumps by up to `jitter` voxels from column to column and has holes, as the stacked 2D traces do"
    rng = np.random.default_rng(seed)
    offsets = rng.integers(-jitter, jitter + 1, size=(SIZE, SIZE))
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    for y in range(SIZE):
        for z in range(SIZE):
            if rng.random() > missing:
                mask[PLANE_X + offsets[y, z], y, z] = True
    return mask


def test_a_ragged_plane_with_holes_becomes_one_voxel_per_column():
    sheet = oriented_sheet(ragged_plane())
    per_column = sheet[:, INTERIOR, INTERIOR].sum(0)
    assert (per_column == 1).all()


def test_the_sheet_stays_next_to_where_the_plane_was_drawn():
    mask = ragged_plane()
    sheet = oriented_sheet(mask)
    distance_to_mask = ndimage.distance_transform_edt(~mask)
    assert distance_to_mask[sheet].max() <= 2


def test_a_lone_voxel_far_from_the_plane_is_dropped():
    mask = ragged_plane()
    mask[PLANE_X + 12, 3, 3] = True
    assert not oriented_sheet(mask)[PLANE_X + 12, 3, 3]


def test_a_thin_line_is_dropped_because_it_is_not_a_sheet():
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    mask[PLANE_X, 5:35, 20] = True
    assert not oriented_sheet(mask).any()


def test_an_empty_mask_gives_an_empty_sheet():
    assert not oriented_sheet(np.zeros((6, 6, 6), dtype=bool)).any()


def test_a_tilted_plane_stays_one_voxel_thick():
    y, z = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    x = np.round(10 + 0.5 * y).astype(int)
    mask = np.zeros((SIZE + 20, SIZE, SIZE), dtype=bool)
    mask[x, y, z] = True
    mask[x[::2, ::3], y[::2, ::3], z[::2, ::3]] = True
    mask &= np.random.default_rng(1).random(mask.shape) > 0.15
    sheet = oriented_sheet(mask)
    thickness = sheet[:, INTERIOR, INTERIOR].sum(0)
    assert thickness.max() <= 2 and (thickness >= 1).mean() > 0.95
