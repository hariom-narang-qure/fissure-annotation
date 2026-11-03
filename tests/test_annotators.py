import re

import base64
import io

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from fissure_annotation.annotators import StackAnnotator, stacks_of
from fissure_annotation.fissure_mask import FissureMaskJob, make_fissure_mask
from test_agent import FakeModel
from test_fissure_mask import scan_with_a_bright_line

SLICES = 7


def test_stacks_follow_each_other_and_the_last_one_can_be_smaller():
    assert stacks_of([0, 3, 6, 9, 12, 15, 18], 3) == [[0, 3, 6], [9, 12, 15], [18]]


class TraceEverySliceModel:
    "each conversation traces the bright line on every slice of its first message, then finishes"

    def __init__(self):
        self.stacks, self.instructions = [], []

    def create(self, **request):
        if "previous_response_id" not in request:
            labels = [part["text"] for part in request["input"][0]["content"] if part["type"] == "input_text"]
            stack = [int(re.match(r"Slice (\d+):", label).group(1)) for label in labels[1:]]
            self.stacks.append(stack)
            self.instructions.append(request["instructions"])
            fissures = [{"slice": z, "name": "oblique", "points": [[100, 300], [399, 300]]} for z in stack]
            self.model = FakeModel([("trace_points", {"fissures": fissures})], [("finish", {"summary": "x"})])
        return self.model.create(**request)

    @property
    def responses(self):
        return self


@pytest.fixture
def scan(tmp_path):
    scan_with_a_bright_line(tmp_path / "scan.nii.gz", slices=SLICES)
    return tmp_path / "scan.nii.gz"


def stack_job(tmp_path, scan, **settings) -> FissureMaskJob:
    return FissureMaskJob(
        **{
            "nifti": scan,
            "mask_path": tmp_path / "out" / "fissures.nii.gz",
            "workers": 1,
            "annotator": StackAnnotator(size=3),
            **settings,
        }
    )


def test_the_stack_annotator_runs_one_conversation_per_stack_and_masks_every_slice(tmp_path, scan):
    model = TraceEverySliceModel()
    job = stack_job(tmp_path, scan, slice_step=2)
    make_fissure_mask(job, model)
    assert model.stacks == [[0, 2, 4], [6]]
    assert "2.0 mm" in model.instructions[0]  # 1 mm slices, 2 apart
    mask = np.asarray(nib.load(job.mask_path).dataobj)
    assert [z for z in range(SLICES) if mask[:, :, z].any()] == [0, 2, 4, 6]


def test_a_stack_with_a_log_is_not_run_again(tmp_path, scan):
    job = stack_job(tmp_path, scan)
    make_fissure_mask(job, TraceEverySliceModel())
    (job.work_folder / "axial" / "run" / "stack_logs" / "stack_0003_0005.json").unlink()
    again = TraceEverySliceModel()
    make_fissure_mask(job, again)
    assert again.stacks == [[3, 4, 5]]


def test_a_config_picks_the_annotator_by_kind(tmp_path):
    job = FissureMaskJob.model_validate(
        {"nifti": tmp_path / "a.nii.gz", "mask_path": tmp_path / "m.nii.gz", "annotator": {"kind": "stack", "size": 3}}
    )
    assert isinstance(job.annotator, StackAnnotator) and job.annotator.size == 3


def first_picture(request) -> np.ndarray:
    first = next(p for p in request["input"][0]["content"] if p["type"] == "input_image")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(first["image_url"].split(",")[1]))).convert("L"))


def test_the_stack_annotator_shows_the_agent_the_ridge_map_without_guides(tmp_path, scan):
    model = TraceEverySliceModel()
    requests = []
    create = model.create
    model.create = lambda **request: requests.append(request) or create(**request)
    job = stack_job(tmp_path, scan)
    make_fissure_mask(job, model)
    seen = first_picture(requests[0])
    assert (seen == 0).mean() > 0.5 and seen.max() > 100  # a ridge map is black with a bright line, the CT slice is lung grey
    assert "GUIDES" not in model.instructions[0]
    assert np.asarray(nib.load(job.mask_path).dataobj).any()
