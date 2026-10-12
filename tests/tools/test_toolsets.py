import itertools

import pytest

from fissure_annotation.tools.schemas import TOOLS
from fissure_annotation.tools.toolsets import scroll_tools, without_ticks, without_tracer


def names(tools: list[dict]) -> list[str]:
    return [t["name"] for t in tools]


def test_the_scroll_agent_has_look_and_check_points_first():
    assert names(scroll_tools())[:2] == ["look", "check_points"]


@pytest.mark.parametrize("tracer, reviewer, plain", list(itertools.product((True, False), repeat=3)))
def test_submit_is_offered_once(tracer, reviewer, plain):
    assert names(scroll_tools(tracer, reviewer, plain)).count("submit") == 1


def test_every_scroll_tool_requires_the_slice_it_acts_on():
    for tool in scroll_tools():
        if tool["name"] not in ("submit",):
            assert "slice" in tool["parameters"]["required"], tool["name"]


def test_without_the_tracer_snap_becomes_mark_and_geometry_goes_away():
    tools = scroll_tools(use_tracer=False)
    assert "snap" not in names(tools) and "mark" in names(tools)
    submit = next(t for t in tools if t["name"] == "submit")
    line = submit["parameters"]["properties"]["lines"]["items"]
    assert "geometry" not in line["properties"] and "geometry" not in line["required"]


def test_with_the_tracer_geometry_is_a_required_choice():
    submit = next(t for t in scroll_tools() if t["name"] == "submit")
    assert "geometry" in submit["parameters"]["properties"]["lines"]["items"]["required"]


def test_review_flag_only_appears_when_a_reviewer_is_configured():
    overlay = lambda tools: next(t for t in tools if t["name"] == "overlay")["parameters"]["properties"]
    assert "review" in overlay(scroll_tools(use_reviewer=True))
    assert "review" not in overlay(scroll_tools(use_reviewer=False))


def test_plain_mode_never_mentions_tick_marks():
    assert not any("tick" in t["description"] for t in scroll_tools(plain=True))


def test_building_a_toolset_does_not_change_the_shared_schemas():
    before = repr(TOOLS)
    scroll_tools(use_tracer=False, use_reviewer=True, plain=True)
    without_ticks(TOOLS)
    without_tracer(TOOLS)
    assert repr(TOOLS) == before
