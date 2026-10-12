import pytest

from fissure_annotation.config import AgentConfig
from fissure_annotation.scroll_experiment import log_dir_name, results_file_name, variant_tag


@pytest.mark.parametrize("agent, tag", [
    (AgentConfig(), "AGS"),
    (AgentConfig(use_tracer=False), "AGN"),
    (AgentConfig(use_tracer=False, use_reviewer=True), "AGR"),
    (AgentConfig(use_tracer=False, plain_images=True), "AGP"),
    (AgentConfig(plain_images=True, use_reviewer=True), "AGP"),
])
def test_variant_tag_names_the_agent_setup(agent, tag):
    assert variant_tag(agent) == tag


def test_output_names_are_the_ones_earlier_rounds_used():
    assert log_dir_name(AgentConfig()) == "agent_log_scroll"
    assert log_dir_name(AgentConfig(use_tracer=False)) == "agent_log_scroll_notracer"
    assert results_file_name(AgentConfig(use_tracer=False, plain_images=True)) == "results_scroll_plain.json"
