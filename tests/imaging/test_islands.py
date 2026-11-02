import numpy as np

from fissure_annotation.imaging.islands import Islands

SIZE = 30


def blocks(*sizes_and_x):
    "one flat block of n voxels per (n, x), each at its own x so they never touch"
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    for voxels, x in sizes_and_x:
        mask[x, 0, :voxels] = True
    return mask


def test_islands_are_numbered_largest_first():
    islands = Islands(blocks((3, 0), (20, 10), (8, 20)), np.eye(4))
    assert islands.sizes.tolist() == [20, 8, 3]
    assert (islands.labels[10, 0, :20] == 1).all()


def test_diagonal_neighbours_are_one_island():
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    assert Islands(mask, np.eye(4)).count == 1


def test_without_removes_islands_and_renumbers_the_rest():
    islands = Islands(blocks((3, 0), (20, 10), (8, 20)), np.eye(4)).without(np.array([1]))
    assert islands.sizes.tolist() == [8, 3]


def test_centroid_distance_is_in_millimetres():
    mask = blocks((1, 0), (1, 10))
    two_mm_in_x = np.diag([2.0, 1.0, 1.0, 1.0])
    assert Islands(mask, two_mm_in_x).centroid_distance_mm(1, 2) == 20.0


def test_an_empty_mask_has_no_islands():
    assert Islands(np.zeros((SIZE, SIZE, SIZE), dtype=bool), np.eye(4)).count == 0
