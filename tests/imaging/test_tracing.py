import numpy as np

from fissure_annotation.imaging.tracing import trace_mask


def arc_image() -> tuple[np.ndarray, list[int]]:
    "dark lung with a bright 3 px thick arc that bends 8 px away from the straight line between its ends"
    image = np.full((60, 60), 50, dtype=np.uint8)
    rows = [30 + round(8 * np.sin(np.pi * x / 59)) for x in range(60)]
    for x, row in enumerate(rows):
        image[row - 1 : row + 2, x] = 200
    return image, rows


def test_the_trace_follows_the_ridge_between_two_waypoints():
    image, rows = arc_image()
    mask = trace_mask(image, [[(2, rows[2]), (57, rows[57])]])
    traced_rows, traced_columns = np.nonzero(mask)
    assert max(abs(traced_rows - np.array([rows[c] for c in traced_columns]))) <= 1
    assert traced_columns.min() == 2 and traced_columns.max() == 57


def test_the_trace_passes_through_every_waypoint():
    image, rows = arc_image()
    waypoints = [
        (2, rows[2]),
        (30, 20),
        (57, rows[57]),
    ]  # the middle one is off the arc on purpose
    mask = trace_mask(image, [waypoints])
    assert all(mask[y, x] for x, y in waypoints)


def test_no_fissures_gives_an_empty_mask():
    image, _ = arc_image()
    assert not trace_mask(image, []).any()
