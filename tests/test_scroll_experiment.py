import json

import pytest

from fissure_annotation import scroll_experiment
from fissure_annotation.config import AgentConfig
from fissure_annotation.scroll_experiment import ScrollExperimentConfig, jobs_for, log_dir_name, results_file_name, variant_tag


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


@pytest.fixture
def two_cases(tmp_path, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"scan1": ["a.nii.gz", "lobes1"], "scan2": ["b.nii.gz", "lobes2"]}))
    for name in ("scan1", "scan2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "picks.json").write_text(json.dumps([["coronal", 10], ["axial", 20], ["axial", 30]]))
    monkeypatch.setattr(scroll_experiment, "OUT", tmp_path)
    monkeypatch.setattr(scroll_experiment, "load_case", lambda scan, lobes: f"case:{scan}")
    return cases


def test_every_picked_slice_of_every_case_is_a_job(two_cases):
    jobs = jobs_for(ScrollExperimentConfig(cases=two_cases))
    assert [(j[1], j[2], j[3]) for j in jobs][:3] == [("scan1", "coronal", 10), ("scan1", "axial", 20), ("scan1", "axial", 30)]
    assert len(jobs) == 6


def test_only_filters_cases_by_name(two_cases):
    jobs = jobs_for(ScrollExperimentConfig(cases=two_cases, only="scan2"))
    assert {j[1] for j in jobs} == {"scan2"}


def test_max_jobs_gives_a_quick_smoke_run(two_cases):
    assert len(jobs_for(ScrollExperimentConfig(cases=two_cases, max_jobs=2))) == 2
