"""The whole scroll experiment with a scripted model instead of the API: config -> agent loop -> tools -> fissures -> files."""
import json
from types import SimpleNamespace

import pytest

from fissure_annotation import cases, scroll_agent, scroll_experiment
from fissure_annotation.scroll_experiment import ScrollExperimentConfig, run_scroll_experiment
from fissure_annotation.settings import load_settings

START = 30
WAYPOINTS = [{"x": 300, "y": 200}, {"x": 500, "y": 400}, {"x": 700, "y": 600}]


def call(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=call_id)


def response(*calls) -> SimpleNamespace:
    return SimpleNamespace(id="r", output=list(calls), usage=SimpleNamespace(input_tokens=10, output_tokens=5, output_tokens_details=None))


class ScriptedClient:
    "plays back one response per turn and keeps what the agent sent back, so tests can see what the model would have seen"

    def __init__(self, turns):
        self.turns = iter(turns)
        self.sent = []
        self.responses = self

    def create(self, **kwargs):
        self.sent.append(kwargs.get("input"))
        return next(self.turns)


@pytest.fixture
def experiment(tmp_path, monkeypatch, case):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps({"scanA": ["a.nii.gz", "lobes"]}))
    (tmp_path / "scanA").mkdir()
    (tmp_path / "scanA" / "picks.json").write_text(json.dumps([["coronal", START]]))
    for module in (cases, scroll_experiment):
        monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(cases, "load_case", lambda scan, lobes: case)
    monkeypatch.setenv("PORTKEY_API_KEY", "not-a-real-key")
    client = ScriptedClient([
        response(call("mark", {"slice": START, "name": "oblique", "side": "right", "waypoints": WAYPOINTS}, "call-mark")),
        response(call("submit", {"lines": [{"line_id": 1, "status": "confirmed", "evidence": "a thin bright line"}]}, "call-submit")),
    ])
    monkeypatch.setattr(scroll_agent, "Portkey", lambda api_key: client)
    config = load_settings(ScrollExperimentConfig, ["--cases", str(cases_file), "--workers", "1", "--agent.use-tracer", "false"])
    return tmp_path, config, client


def test_one_slice_runs_from_config_to_files(experiment):
    out, config, _ = experiment
    (row,) = run_scroll_experiment(config)
    assert (row["case"], row["plane"], row["index"], row["variant"]) == ("scanA", "coronal", START, "AGN")
    assert row["n_fissures"] == 1 and row["steps"] == 1
    assert row["tokens"] == {"input": 20, "output": 10, "reasoning": 0}


def test_the_delivered_fissure_is_what_the_model_marked_and_submitted(experiment):
    out, config, _ = experiment
    run_scroll_experiment(config)
    (fissure,) = json.loads((out / "scanA" / f"coronal_{START}" / "AGN.json").read_text())
    assert (fissure["name"], fissure["side"], fissure["confidence"], fissure["geometry"]) == ("oblique", "right", "clear", "mine")
    assert fissure["waypoints"] == WAYPOINTS
    assert fissure["evidence"] == "a thin bright line"


def test_review_files_and_logs_are_written(experiment):
    out, config, _ = experiment
    run_scroll_experiment(config)
    folder = out / "scanA" / f"coronal_{START}"
    assert (folder / "AGN.panel.png").exists() and (folder / "AGN_extra.json").exists()
    assert (folder / "agent_log_scroll_notracer" / "transcript.json").exists()
    assert json.loads((out / "results_scroll_notracer.json").read_text())[0]["n_fissures"] == 1


def test_the_model_is_shown_the_line_it_drew(experiment):
    _, config, client = experiment
    run_scroll_experiment(config)
    tool_output = client.sent[1][0]["output"]
    assert "line_id" in tool_output[0]["text"] and tool_output[1]["type"] == "input_image"


def test_a_tool_error_is_reported_to_the_model_and_the_run_continues(experiment, monkeypatch):
    out, config, _ = experiment
    bad = ScriptedClient([
        response(call("mark", {"slice": 9999, "name": "oblique", "side": "right", "waypoints": WAYPOINTS}, "c1")),
        response(call("submit", {"lines": []}, "c2")),
    ])
    monkeypatch.setattr(scroll_agent, "Portkey", lambda api_key: bad)
    (row,) = run_scroll_experiment(config)
    assert row["n_fissures"] == 0
    assert "outside the lung range" in bad.sent[1][0]["output"][0]["text"]
