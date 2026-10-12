import inspect
import itertools

import pytest

from fissure_annotation.agent import Slice
from fissure_annotation.slicing import cut
from fissure_annotation.tools.toolsets import scroll_tools

HANDLED_BEFORE_THE_SLICE = {"slice", "completed", "review"}
"""Arguments the `Stack` strips before it calls the slice's method: which slice, whether to also return the completed image, and whether to ask the reviewer."""
HANDLED_BY_THE_CALLER = {"look", "submit"}

SETUPS = list(itertools.product((True, False), (True, False), (True, False)))


@pytest.fixture
def slice_factory(case):
    return lambda tracer: Slice(cut(case, "coronal", 30), use_tracer=tracer)


@pytest.mark.parametrize("tracer, reviewer, plain", SETUPS)
def test_every_tool_the_model_is_offered_has_an_implementation(slice_factory, tracer, reviewer, plain):
    methods = slice_factory(tracer).tool_methods()
    offered = {t["name"] for t in scroll_tools(tracer, reviewer, plain)} - HANDLED_BY_THE_CALLER
    assert offered == set(methods)


@pytest.mark.parametrize("tracer", [True, False])
def test_every_schema_argument_is_one_the_method_accepts(slice_factory, tracer):
    methods = slice_factory(tracer).tool_methods()
    for tool in scroll_tools(tracer, use_reviewer=True):
        if tool["name"] in HANDLED_BY_THE_CALLER:
            continue
        accepted = set(inspect.signature(methods[tool["name"]]).parameters)
        offered = set(tool["parameters"]["properties"]) - HANDLED_BEFORE_THE_SLICE
        assert offered <= accepted, f"{tool['name']}: {offered - accepted}"


@pytest.mark.parametrize("tracer", [True, False])
def test_every_required_argument_without_a_default_is_in_the_schema(slice_factory, tracer):
    methods = slice_factory(tracer).tool_methods()
    for tool in scroll_tools(tracer):
        if tool["name"] in HANDLED_BY_THE_CALLER:
            continue
        params = inspect.signature(methods[tool["name"]]).parameters
        must_pass = {n for n, p in params.items() if p.default is inspect.Parameter.empty}
        assert must_pass <= set(tool["parameters"]["required"]), tool["name"]
