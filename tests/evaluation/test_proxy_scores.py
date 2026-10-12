import numpy as np

from fissure_annotation.agent import Slice
from fissure_annotation.evaluation.proxy_scores import polyline_points, proxy_scores
from fissure_annotation.lines import ProposedFissure, Waypoint
from fissure_annotation.slicing import cut


def boundary_column(sl: Slice) -> int:
    "first column, on the middle row, where two different lobes touch"
    row = sl.view.lobes[sl.view.lobes.shape[0] // 2]
    touching = (row[:-1] != row[1:]) & (row[:-1] > 0) & (row[1:] > 0)
    return int(np.nonzero(touching)[0][0])


def vertical_fissure_at(sl: Slice, column: int) -> ProposedFissure:
    x = sl.to_model(column, 0)[0]
    return ProposedFissure(name="oblique", side="right", confidence="clear", waypoints=[Waypoint(x=x, y=300), Waypoint(x=x, y=700)])


def test_polyline_points_are_spaced_along_the_line_and_keep_both_ends():
    pts = polyline_points([(0.0, 0.0), (10.0, 0.0)], step=2.0)
    assert pts[0].tolist() == [0.0, 0.0] and pts[-1].tolist() == [10.0, 0.0]
    assert np.allclose(np.diff(pts[:, 0]), 2.0)


def test_no_fissures_means_no_score(case):
    sl = Slice(cut(case, "coronal", 30), use_tracer=False)
    assert proxy_scores(sl.view, [], sl.prepared) == {"dist_mm": None, "within5": None, "recall5": None}


def test_a_line_on_the_lobe_boundary_scores_better_than_one_away_from_it(case):
    sl = Slice(cut(case, "axial", 30), use_tracer=False)
    column = boundary_column(sl)
    near = proxy_scores(sl.view, [vertical_fissure_at(sl, column)], sl.prepared)
    far = proxy_scores(sl.view, [vertical_fissure_at(sl, column + 12)], sl.prepared)
    assert near["dist_mm"] < far["dist_mm"]
    assert near["within5"] > far["within5"]
