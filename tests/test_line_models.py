import numpy as np
import pytest
from pydantic import ValidationError

from fissure_annotation.agent import Slice
from fissure_annotation.lines import Fissure, Line, MainPoint, ProposedFissure, SubmittedLine, Waypoint
from fissure_annotation.slicing import cut

POINTS = [{"x": 300, "y": 120}, {"x": 500, "y": 300}, {"x": 700, "y": 500}]


@pytest.fixture
def slice_without_tracer(case) -> Slice:
    return Slice(cut(case, "coronal", 30), use_tracer=False)


def verdict(line_id: int, status: str, geometry: str = "mine") -> dict:
    return {"line_id": line_id, "status": status, "geometry": geometry, "evidence": "e"}


def test_a_line_without_a_trace_is_not_traced():
    line = Line(name="oblique", side="right", own_points=[MainPoint(x=1, y=2)])
    assert not line.is_traced


def test_a_traced_path_must_be_n_by_2():
    with pytest.raises(ValidationError):
        Line(name="oblique", side="right", own_points=[], traced_path=np.zeros((5, 3)))


def test_names_and_sides_outside_the_vocabulary_are_rejected():
    with pytest.raises(ValidationError):
        Line(name="oblique", side="both", own_points=[])


def test_submitted_geometry_defaults_to_the_models_own_points():
    assert SubmittedLine(line_id=1, status="confirmed", evidence="e").geometry == "mine"


def test_confirmed_is_clear_and_uncertain_is_faint(slice_without_tracer):
    sl = slice_without_tracer
    sl.mark_tool(POINTS, "oblique", "right")
    sl.mark_tool(POINTS, "horizontal", "left")
    clear, faint = sl.final_fissures([verdict(1, "confirmed"), verdict(2, "uncertain")])
    assert (clear.confidence, faint.confidence) == ("clear", "faint")


def test_dropped_and_unknown_lines_are_not_delivered(slice_without_tracer):
    sl = slice_without_tracer
    sl.mark_tool(POINTS, "oblique", "right")
    assert sl.final_fissures([verdict(1, "dropped"), verdict(99, "confirmed")]) == []


def test_waypoints_are_whole_numbers_in_the_main_frame(slice_without_tracer):
    sl = slice_without_tracer
    sl.mark_tool(POINTS, "oblique", "right")
    (fissure,) = sl.final_fissures([verdict(1, "confirmed")])
    assert [w.model_dump() for w in fissure.waypoints] == POINTS
    assert fissure.geometry == "mine"


def test_traced_geometry_is_ignored_for_a_line_that_was_never_traced(slice_without_tracer):
    sl = slice_without_tracer
    sl.mark_tool(POINTS, "oblique", "right")
    (fissure,) = sl.final_fissures([verdict(1, "confirmed", geometry="traced")])
    assert fissure.geometry == "mine"


def test_dumping_a_fissure_keeps_the_key_order_of_the_json_files():
    sl_fissure = Fissure(name="oblique", side="right", confidence="clear", waypoints=[Waypoint(x=1, y=2)], evidence="e", geometry="mine", fit={})
    assert list(sl_fissure.model_dump()) == ["name", "side", "confidence", "waypoints", "evidence", "geometry", "fit"]


def test_a_one_shot_reply_with_decimal_waypoints_is_kept_as_given():
    f = ProposedFissure.model_validate({"name": "oblique", "side": "left", "confidence": "uncertain", "waypoints": [{"x": 10.5, "y": 20}]})
    assert (f.waypoints[0].x, f.waypoints[0].y) == (10.5, 20)
    assert isinstance(f.waypoints[0].y, int)


def test_a_one_shot_fissure_with_an_unknown_side_word_is_rejected():
    with pytest.raises(ValidationError):
        ProposedFissure.model_validate({"name": "oblique", "side": "both", "confidence": "clear", "waypoints": []})
