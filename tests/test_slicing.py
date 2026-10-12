import numpy as np

from fissure_annotation.slicing import boundary_mask, cut, lung_window, orient


def test_patient_right_is_image_left_and_superior_is_up():
    a = np.zeros((4, 5))
    a[3, 4] = 1  # highest x (patient right in RAS) and highest z (superior)
    assert orient(a)[0, 0] == 1


def test_coronal_cut_puts_patient_right_on_image_left(case):
    case.vol[50:58, 30, :] = 0.0  # a block, since resampling interpolates a single voxel away
    view = cut(case, "coronal", 30)
    _, cols = np.nonzero(view.hu > -400)
    assert cols.max() < view.hu.shape[1] // 4


def test_pixels_are_square_at_the_coarser_spacing(case):
    view = cut(case, "coronal", 30)
    assert view.mm_per_px == 2.0
    assert view.image.shape == (60, 30)  # z kept at 2 mm, x (1 mm) halved


def test_lobes_follow_the_same_flip_as_the_image(case):
    view = cut(case, "axial", 30)
    assert view.lobes.shape == view.image.shape
    right_lung_columns = np.nonzero((view.lobes == 3).any(axis=0))[0]
    left_lung_columns = np.nonzero((view.lobes == 1).any(axis=0))[0]
    assert right_lung_columns.max() < left_lung_columns.min()


def test_lung_window_clips_both_ends():
    out = lung_window(np.array([-5000.0, 5000.0]))
    assert list(out) == [0, 255]


def test_boundary_mask_marks_only_touching_lobes():
    lobes = np.zeros((6, 6), dtype=np.uint8)
    lobes[:, :3] = 1
    lobes[:, 3:] = 2
    mask = boundary_mask(lobes)
    assert mask[:, 2:4].all()
    assert not mask[:, :2].any() and not mask[:, 4:].any()


def test_boundary_mask_ignores_lobe_against_background():
    lobes = np.zeros((6, 6), dtype=np.uint8)
    lobes[:, :3] = 1
    assert not boundary_mask(lobes).any()
