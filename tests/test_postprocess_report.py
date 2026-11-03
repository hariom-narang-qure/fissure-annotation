import numpy as np

from fissure_annotation.postprocess_report import RemovedVoxels


def voxels(*xs):
    mask = np.zeros((4, 4, 4), dtype=bool)
    for x in xs:
        mask[x] = True
    return mask


def test_each_step_gets_the_next_label_from_one():
    removed = RemovedVoxels((4, 4, 4))
    removed.add("first", voxels(0))
    removed.add("second", voxels(2))
    assert (removed.labels[0] == 1).all() and (removed.labels[2] == 2).all()
    assert (removed.labels[1] == 0).all()


def test_a_voxel_removed_twice_keeps_the_label_of_the_first_step():
    removed = RemovedVoxels((4, 4, 4))
    removed.add("first", voxels(0))
    removed.add("second", voxels(0, 1))
    assert (removed.labels[0] == 1).all()
    assert [step.voxels for step in removed.steps] == [16, 16]


def test_a_step_that_removed_nothing_is_still_in_the_legend():
    removed = RemovedVoxels((4, 4, 4))
    removed.add("nothing", voxels())
    assert [(step.label, step.step, step.voxels) for step in removed.steps] == [(1, "nothing", 0)]
