import numpy as np

from fissure_annotation.tracing import arc_length, ridge_map, smooth, snap

SIZE = 120
LINE_ROW = 60


def slice_with_a_bright_line() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((SIZE, SIZE), 40, dtype=np.uint8)
    image[LINE_ROW, 15:105] = 200
    mask = np.ones((SIZE, SIZE), dtype=bool)
    return image, mask


def test_ridge_map_is_zero_outside_the_lung():
    image, mask = slice_with_a_bright_line()
    mask[:, :30] = False
    assert ridge_map(image, mask)[:, :20].max() == 0


def test_snap_pulls_offset_waypoints_onto_the_line():
    image, mask = slice_with_a_bright_line()
    score = ridge_map(image, mask)
    offset = [(LINE_ROW + 3, 30), (LINE_ROW - 3, 60), (LINE_ROW + 3, 90)]
    result = snap(score, offset, mm_per_px=1.0)
    middle = result.path[len(result.path) // 2]
    assert abs(middle[0] - LINE_ROW) <= 1.5
    assert result.stats["fraction_on_ridge"] > 0.8


def test_snap_leaves_waypoints_in_empty_lung_where_they_are():
    image = np.full((SIZE, SIZE), 40, dtype=np.uint8)
    score = ridge_map(image, np.ones((SIZE, SIZE), dtype=bool))
    result = snap(score, [(30, 30), (30, 90)], mm_per_px=1.0)
    assert np.allclose(result.path[0], (30, 30)) and np.allclose(result.path[-1], (30, 90))


def test_smoothing_keeps_the_end_points():
    path = np.array([[0.0, 0.0], [5.0, 9.0], [0.0, 18.0], [5.0, 27.0], [0.0, 36.0]])
    out = smooth(path, sigma_px=2.0)
    assert np.allclose(out[0], path[0]) and np.allclose(out[-1], path[-1])
    assert arc_length(out) < arc_length(path)
