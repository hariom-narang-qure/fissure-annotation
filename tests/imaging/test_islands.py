import numpy as np

from fissure_annotation.imaging.islands import BACKGROUND, Islands

SIZE = 30


def blocks(*sizes_and_x):
    "one flat block of n voxels per (n, x), each at its own x so they never touch"
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    for voxels, x in sizes_and_x:
        mask[x, 0, :voxels] = True
    return mask


def test_islands_are_ranked_largest_first():
    islands = Islands(blocks((3, 0), (20, 10), (8, 20)), np.eye(4))
    assert islands.sizes.tolist() == [20, 8, 3]
    assert (islands.ranks[10, 0, :20] == 0).all()
    assert (islands.ranks[0, 0, :3] == 2).all()
    assert (islands.ranks[5] == BACKGROUND).all()


def test_diagonal_neighbours_are_one_island():
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    assert Islands(mask, np.eye(4)).count == 1


def test_without_removes_islands_and_reranks_the_rest():
    islands = Islands(blocks((3, 0), (20, 10), (8, 20)), np.eye(4)).without(np.array([0]))
    assert islands.sizes.tolist() == [8, 3]


def test_centroid_distance_is_in_millimetres():
    mask = blocks((1, 0), (1, 10))
    two_mm_in_x = np.diag([2.0, 1.0, 1.0, 1.0])
    assert Islands(mask, two_mm_in_x).centroid_distance_mm(0, 1) == 20.0


def test_an_empty_mask_has_no_islands():
    assert Islands(np.zeros((SIZE, SIZE, SIZE), dtype=bool), np.eye(4)).count == 0
