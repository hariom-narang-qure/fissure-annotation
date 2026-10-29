import numpy as np
import pytest

from fissure_annotation.agent import AgentSettings
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.stack_agent import BandMaskImages, BandOutlines, StackFolder, run_stack
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


def test_with_hints_each_slice_is_followed_by_its_band_mask_and_zooms_crop_it_too(slices_dir, tmp_path):
    hints_dir = tmp_path / "hints"
    for z in STACK:
        save_png(np.zeros((512, 512), dtype=np.uint8), hints_dir / f"slice_{z:04d}.png")
    model = FakeModel(
        [("zoom", {"slice": 8, "x0": 0, "y0": 200, "x1": 200, "y1": 400, "scale": 2})],
        [("finish", {"summary": "x"})],
    )
    folder = StackFolder(root=tmp_path / "run")
    masks = {z: read_gray(hints_dir / f"slice_{z:04d}.png") for z in STACK}
    run_stack(model, AgentSettings(), folder, slices_dir, STACK, step_mm=3.2, hints=BandMaskImages(masks))
    content = model.requests[0]["input"][0]["content"]
    labels = [part["text"] for part in content if part["type"] == "input_text"]
    assert labels[1:3] == ["Slice 4:", "Band mask of slice 4:"]
    assert sum(part["type"] == "input_image" for part in content) == 6
    zoom_reply = model.requests[1]["input"][0]["output"]
    assert sum(part["type"] == "input_image" for part in zoom_reply) == 2


def test_with_outlines_the_slices_are_drawn_on_and_the_trace_still_follows_the_clean_slice(slices_dir, tmp_path):
    "a band whose edge sits on the bright line must not change where the trace goes, only what the model sees"
    masks = {}
    for z in STACK:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[300 + z - 5 : 300 + z + 6, 100:400] = 170
        masks[z] = mask
    model = FakeModel([("trace_points", {"fissures": [line_on(z) for z in STACK]})], [("finish", {"summary": "x"})])
    folder = StackFolder(root=tmp_path / "run")
    run_stack(model, AgentSettings(), folder, slices_dir, STACK, step_mm=3.2, hints=BandOutlines(masks, {170: (255, 220, 0)}))
    content = model.requests[0]["input"][0]["content"]
    assert sum(part["type"] == "input_image" for part in content) == 3  # no extra mask images
    traced = read_gray(folder.mask_path(slices_dir / "slice_0008.png")) > 0
    assert set(np.nonzero(traced)[0]) == {308}


def test_overview_outlines_are_drawn_only_in_the_first_message(slices_dir, tmp_path):
    import base64
    import io

    from PIL import Image

    from fissure_annotation.stack_agent import OverviewBandOutlines

    def pixels(part):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(part["image_url"].split(",")[1]))).convert("RGB"))

    def has_yellow(picture):
        return ((picture == (255, 220, 0)).all(axis=-1)).any()

    masks = {z: np.zeros((512, 512), dtype=np.uint8) for z in STACK}
    for z in STACK:
        masks[z][250:290, 100:400] = 170
    model = FakeModel(
        [("zoom", {"slice": 8, "x0": 90, "y0": 240, "x1": 410, "y1": 300, "scale": 1})],
        [("finish", {"summary": "x"})],
    )
    hints = OverviewBandOutlines(masks, {170: (255, 220, 0)})
    run_stack(model, AgentSettings(), StackFolder(root=tmp_path / "run"), slices_dir, STACK, step_mm=3.2, hints=hints)
    first_images = [p for p in model.requests[0]["input"][0]["content"] if p["type"] == "input_image"]
    zoom_image = model.requests[1]["input"][0]["output"][0]
    assert has_yellow(pixels(first_images[0]))
    assert not has_yellow(pixels(zoom_image))
