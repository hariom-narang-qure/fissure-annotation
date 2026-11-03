from types import SimpleNamespace

import numpy as np
import pytest
from test_agent import FakeModel
from test_stack_agent import STACK, line_on

from fissure_annotation.agent import AgentSettings, ReplayedTurns
from fissure_annotation.experiment import OPENROUTER_BASE_URL, model_client
from fissure_annotation.imaging.png import save_png
from fissure_annotation.retry import is_out_of_credit, is_transient
from fissure_annotation.stack_agent import StackFolder, run_stack

OPENROUTER = AgentSettings(provider="openrouter", model="openai/gpt-6.1-terra")


@pytest.fixture
def slices_dir(tmp_path):
    for z in STACK:
        save_png(np.full((512, 512), 50, dtype=np.uint8), tmp_path / "slices" / f"slice_{z:04d}.png")
    return tmp_path / "slices"


def run_on(settings, model, slices_dir, tmp_path):
    return run_stack(model, settings, StackFolder(root=tmp_path / "run"), slices_dir, STACK, step_mm=3.2)


def test_openrouter_gets_the_whole_conversation_every_turn(slices_dir, tmp_path):
    "OpenRouter keeps no responses and refuses previous_response_id"
    model = FakeModel([("trace_points", {"fissures": [line_on(z) for z in STACK]})], [("finish", {"summary": "three"})])
    run_on(OPENROUTER, model, slices_dir, tmp_path)
    first, second = model.requests
    assert "previous_response_id" not in second
    assert second["input"][0] == first["input"][0]
    call, output = second["input"][1:]
    assert (call["type"], call["call_id"]) == ("function_call", "call0")
    assert (output["type"], output["call_id"]) == ("function_call_output", "call0")


def test_portkey_still_chains_the_responses(slices_dir, tmp_path):
    model = FakeModel([("trace_points", {"fissures": [line_on(z) for z in STACK]})], [("finish", {"summary": "three"})])
    run_on(AgentSettings(), model, slices_dir, tmp_path)
    assert model.requests[1]["previous_response_id"] == "r1"
    assert [item["type"] for item in model.requests[1]["input"]] == ["function_call_output"]


def test_reasoning_is_left_out_of_the_resent_conversation():
    "reasoning items can only be sent back to a provider that stored them"
    turns = ReplayedTurns([{"role": "user", "content": "slice"}])
    response = SimpleNamespace(output=[SimpleNamespace(type="reasoning", id="rs1"), SimpleNamespace(type="function_call", call_id="c1")])
    request = turns.next(response, [{"type": "function_call_output", "call_id": "c1", "output": "ok"}])
    assert [item.get("type", "message") for item in request["input"]] == ["message", "function_call", "function_call_output"]


def test_an_openrouter_model_name_has_no_at():
    with pytest.raises(ValueError, match="openai/gpt-6.1-terra"):
        AgentSettings(provider="openrouter", model="@openai/gpt-6.1-terra")


def test_openrouter_out_of_credit_is_waited_out_not_retried():
    "OpenRouter answers 402 Payment Required when its credits run out"
    error = SimpleNamespace(status_code=402)
    assert is_out_of_credit(error) and not is_transient(error)


def test_the_openrouter_client_points_at_openrouter_and_leaves_retries_to_retry_py(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = model_client(OPENROUTER)
    assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert client.max_retries == 0
