import numpy as np
import pytest

from fissure_annotation.agent import AgentSettings
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.stack_agent import OverviewGuides, StackFolder, run_stack
from test_agent import FakeModel

STACK = [4, 8, 12]


@pytest.fixture
def slices_dir(tmp_path):
    for z in STACK:
        image = np.full((512, 512), 50, dtype=np.uint8)
        image[300 + z, 100:400] = 220  # a bright line that moves down one row per slice
        save_png(image, tmp_path / "slices" / f"slice_{z:04d}.png")
    return tmp_path / "slices"


def line_on(z):
    return {"slice": z, "name": "oblique", "points": [[100, 300 + z], [399, 300 + z]]}


def run(model, slices_dir, tmp_path):
    folder = StackFolder(root=tmp_path / "run")
    return folder, run_stack(model, AgentSettings(), folder, slices_dir, STACK, step_mm=3.2)


def test_the_first_message_shows_every_slice_of_the_stack_in_order(slices_dir, tmp_path):
    model = FakeModel([("finish", {"summary": "none"})])
    run(model, slices_dir, tmp_path)
    content = model.requests[0]["input"][0]["content"]
    labels = [part["text"] for part in content if part["type"] == "input_text"]
    assert labels[1:] == ["Slice 4:", "Slice 8:", "Slice 12:"]
    assert sum(part["type"] == "input_image" for part in content) == 3
    assert "3 nearby axial slices" in model.requests[0]["instructions"] and "3.2 mm" in model.requests[0]["instructions"]


def test_one_trace_call_saves_a_mask_per_slice_and_one_log(slices_dir, tmp_path):
    model = FakeModel([("trace_points", {"fissures": [line_on(z) for z in STACK]})], [("finish", {"summary": "three"})])
    folder, log = run(model, slices_dir, tmp_path)
    for z in STACK:
        assert folder.mask_path(slices_dir / f"slice_{z:04d}.png").exists()
    assert log.traces == 1 and log.summary == "three"
    assert folder.stack_log_path(STACK).exists()
    replies = model.requests[1]["input"][0]["output"]
    assert sum(part["type"] == "input_image" for part in replies) == 3


def test_a_later_trace_without_a_slice_empties_that_slice(slices_dir, tmp_path):
    model = FakeModel(
        [("trace_points", {"fissures": [line_on(z) for z in STACK]})],
        [("trace_points", {"fissures": [line_on(4), line_on(8)]})],
        [("finish", {"summary": "two"})],
    )
    folder, _ = run(model, slices_dir, tmp_path)
    png = slices_dir / "slice_0012.png"
    assert not folder.mask_path(png).exists()
    assert folder.load_waypoints(png).fissures == []


def test_a_slice_outside_the_stack_is_refused_and_nothing_is_saved(slices_dir, tmp_path):
    model = FakeModel([("trace_points", {"fissures": [line_on(4), line_on(99)]})], [("finish", {"summary": "x"})])
    folder, log = run(model, slices_dir, tmp_path)
    reply = model.requests[1]["input"][0]["output"]
    assert "not in the stack" in reply[0]["text"]
    assert log.traces == 0 and not folder.mask_path(slices_dir / "slice_0004.png").exists()


def test_zoom_needs_a_slice_of_the_stack(slices_dir, tmp_path):
    model = FakeModel(
        [("zoom", {"slice": 8, "x0": 0, "y0": 200, "x1": 200, "y1": 400, "scale": 2})],
        [("finish", {"summary": "x"})],
    )
    run(model, slices_dir, tmp_path)
    assert model.requests[1]["input"][0]["output"][0]["type"] == "input_image"


def test_guides_are_drawn_on_the_first_message_only_and_the_trace_follows_the_clean_slice(slices_dir, tmp_path):
    import base64
    import io

    from PIL import Image

    def pixels(part):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(part["image_url"].split(",")[1]))).convert("RGB"))

    def has_yellow(picture):
        return ((picture == (255, 220, 0)).all(axis=-1)).any()

    masks = {z: np.zeros((512, 512), dtype=np.uint8) for z in STACK}
    for z in STACK:
        masks[z][300 + z - 5 : 300 + z + 6, 100:400] = 2  # a band whose edges sit right next to the bright line
    model = FakeModel(
        [("zoom", {"slice": 8, "x0": 90, "y0": 280, "x1": 410, "y1": 340, "scale": 1})],
        [("trace_points", {"fissures": [line_on(z) for z in STACK]})],
        [("finish", {"summary": "x"})],
    )
    guides = OverviewGuides(masks, {2: (255, 220, 0)})
    folder = StackFolder(root=tmp_path / "run")
    run_stack(model, AgentSettings(), folder, slices_dir, STACK, step_mm=3.2, guides=guides)
    first_images = [p for p in model.requests[0]["input"][0]["content"] if p["type"] == "input_image"]
    zoom_image = model.requests[1]["input"][0]["output"][0]
    assert has_yellow(pixels(first_images[0]))
    assert not has_yellow(pixels(zoom_image))
    overlays = [p for p in model.requests[2]["input"][0]["output"] if p["type"] == "input_image"]
    assert not any(has_yellow(pixels(p)) for p in overlays)
    traced = read_gray(folder.mask_path(slices_dir / "slice_0008.png")) > 0
    assert set(np.nonzero(traced)[0]) == {308}
