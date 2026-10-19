import json
from types import SimpleNamespace

import numpy as np
import pytest

from fissure_annotation.agent import AgentSettings, run_slice
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.waypoints import RunFolder


class FakeModel:
    "answers each request with the next scripted list of tool calls, and keeps what it was sent"

    def __init__(self, *turns):
        self.turns, self.requests = list(turns), []
        self.responses = self

    def create(self, **request):
        self.requests.append(request)
        calls = [
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=json.dumps(arguments),
                call_id=f"call{i}",
            )
            for i, (name, arguments) in enumerate(self.turns.pop(0))
        ]
        return SimpleNamespace(
            id=f"r{len(self.requests)}",
            output=calls,
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
        )


@pytest.fixture
def slices_dir(tmp_path):
    image = np.full((512, 512), 50, dtype=np.uint8)
    image[300, 100:400] = 220  # a bright horizontal line
    save_png(image, tmp_path / "slices" / "slice_0007.png")
    return tmp_path / "slices"


TRACE = (
    "trace_points",
    {
        "fissures": [{"name": "oblique", "points": [[100, 300], [399, 300]]}],
        "notes": "clear",
    },
)
FINISH = ("finish", {"summary": "one fissure"})


def run(model, slices_dir, tmp_path, **settings):
    return run_slice(
        model,
        AgentSettings(**settings),
        RunFolder(root=tmp_path / "run"),
        slices_dir,
        7,
    )


def tool_outputs(model, request_index):
    return model.requests[request_index]["input"]


def test_a_zoom_trace_finish_conversation_saves_points_mask_overlay_and_log(
    slices_dir, tmp_path
):
    model = FakeModel(
        [("zoom", {"x0": 0, "y0": 200, "x1": 200, "y1": 400, "scale": 3})],
        [TRACE],
        [FINISH],
    )
    log = run(model, slices_dir, tmp_path)
    folder = RunFolder(root=tmp_path / "run")
    png = slices_dir / "slice_0007.png"
    assert log.summary == "one fissure" and log.traces == 1 and log.steps == 3
    assert folder.load_waypoints(png).fissures[0].points == [(100, 300), (399, 300)]
    assert read_gray(folder.mask_path(png))[300, 250] == 255
    assert folder.overlay_path(png).exists() and folder.log_path(png).exists()


def test_a_zoom_comes_back_to_the_model_as_an_image(slices_dir, tmp_path):
    model = FakeModel(
        [("zoom", {"x0": 0, "y0": 200, "x1": 200, "y1": 400, "scale": 3})], [FINISH]
    )
    run(model, slices_dir, tmp_path)
    assert tool_outputs(model, 1)[0]["output"][0]["type"] == "input_image"


def test_the_fourth_trace_is_refused(slices_dir, tmp_path):
    model = FakeModel([TRACE], [TRACE], [TRACE], [TRACE], [FINISH])
    log = run(model, slices_dir, tmp_path)
    assert "limit (3) reached" in tool_outputs(model, 4)[0]["output"][0]["text"]
    assert log.traces == 3


def test_a_bad_tool_call_is_reported_to_the_model_not_raised(slices_dir, tmp_path):
    model = FakeModel(
        [("trace_points", {"fissures": [{"name": "oblique", "points": [[1, 1]]}]})],
        [FINISH],
    )
    run(model, slices_dir, tmp_path)
    assert tool_outputs(model, 1)[0]["output"][0]["text"].startswith("tool error")


def test_no_fissure_is_a_valid_answer(slices_dir, tmp_path):
    log = run(FakeModel([FINISH]), slices_dir, tmp_path)
    assert (
        log.traces == 0
        and not RunFolder(root=tmp_path / "run")
        .points_path(slices_dir / "slice_0007.png")
        .exists()
    )


def test_the_conversation_stops_at_the_step_limit(slices_dir, tmp_path):
    model = FakeModel(
        *[[("zoom", {"x0": 0, "y0": 0, "x1": 50, "y1": 50, "scale": 2})]] * 5
    )
    assert run(model, slices_dir, tmp_path, max_steps=3).steps == 3


class FlakyModel(FakeModel):
    "fails the first request with a 503, like a provider connection timeout"

    def create(self, **request):
        if not getattr(self, "failed", False):
            self.failed = True
            error = Exception("upstream connect error")
            error.status_code = 503
            raise error
        return super().create(**request)


def test_a_transient_provider_error_is_retried_and_the_slice_finishes(slices_dir, tmp_path):
    model = FlakyModel([TRACE], [FINISH])
    log = run(model, slices_dir, tmp_path, retry_wait_seconds=0.0)
    assert log.summary == "one fissure" and log.traces == 1


def test_the_service_tier_is_sent_only_when_it_is_set(slices_dir, tmp_path):
    default, flex = FakeModel([FINISH]), FakeModel([FINISH])
    run(default, slices_dir, tmp_path / "a")
    run(flex, slices_dir, tmp_path / "b", service_tier="flex")
    assert "service_tier" not in default.requests[0]
    assert flex.requests[0]["service_tier"] == "flex"


class NoFlexCapacityModel(FakeModel):
    "answers the first two requests with a lack of flex capacity, as OpenAI does"

    def create(self, **request):
        self.refusals = getattr(self, "refusals", 0)
        if self.refusals < 2:
            self.refusals += 1
            error = Exception("Flex processing is temporarily unavailable")
            error.status_code, error.code = 429, "flex_unavailable"
            raise error
        return super().create(**request)


def test_a_slice_waits_for_flex_capacity_and_says_so(slices_dir, tmp_path, capsys):
    log = run(NoFlexCapacityModel([FINISH]), slices_dir, tmp_path, capacity_wait_seconds=0.0)
    assert log.summary == "one fissure"
    assert "axial slice 7: no flex capacity, wait 2 in 0s" in capsys.readouterr().out
