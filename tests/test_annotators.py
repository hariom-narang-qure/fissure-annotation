import re

import nibabel as nib
import numpy as np
import pytest

from fissure_annotation.annotators import StackAnnotator, stacks_of
from fissure_annotation.fissure_mask import FissureMaskJob, make_fissure_mask
from fissure_annotation.imaging.lobes import LEFT_LOWER, LEFT_UPPER
from fissure_annotation.stack_agent import DEFAULT_PROMPT, NO_GUIDES_PROMPT
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
def scan_and_lobes(tmp_path):
    "the lobes meet at the bright line, so the guides surround it"
    scan_with_a_bright_line(tmp_path / "scan.nii.gz", slices=SLICES)
    lobes = np.zeros((512, 512, SLICES), dtype=np.uint8)
    lobes[100:412, 180:211, :] = LEFT_UPPER  # y 211 is row 300 of the slice, see scan_with_a_bright_line
    lobes[100:412, 211:250, :] = LEFT_LOWER
    nib.save(nib.Nifti1Image(lobes, np.eye(4)), tmp_path / "lobes.nii.gz")
    return tmp_path / "scan.nii.gz", tmp_path / "lobes.nii.gz"


def stack_job(tmp_path, scan, lobes, **settings) -> FissureMaskJob:
    return FissureMaskJob(
        **{
            "nifti": scan,
            "mask_path": tmp_path / "out" / "fissures.nii.gz",
            "lobes": lobes,
            "workers": 1,
            "annotator": StackAnnotator(size=3),
            **settings,
        }
    )


def test_the_stack_annotator_runs_one_conversation_per_stack_and_masks_every_slice(tmp_path, scan_and_lobes):
    scan, lobes = scan_and_lobes
    model = TraceEverySliceModel()
    job = stack_job(tmp_path, scan, lobes, slice_step=2)
    make_fissure_mask(job, model)
    assert model.stacks == [[0, 2, 4], [6]]
    assert "2.0 mm" in model.instructions[0]  # 1 mm slices, 2 apart
    mask = np.asarray(nib.load(job.mask_path).dataobj)
    assert [z for z in range(SLICES) if mask[:, :, z].any()] == [0, 2, 4, 6]
    assert (job.work_folder / "fissure_bands" / "axial" / "slice_0000.png").exists()


def test_a_stack_with_a_log_is_not_run_again(tmp_path, scan_and_lobes):
    scan, lobes = scan_and_lobes
    job = stack_job(tmp_path, scan, lobes)
    make_fissure_mask(job, TraceEverySliceModel())
    (job.work_folder / "axial" / "run" / "stack_logs" / "stack_0003_0005.json").unlink()
    again = TraceEverySliceModel()
    make_fissure_mask(job, again)
    assert again.stacks == [[3, 4, 5]]


def test_the_stack_annotator_needs_lobes_for_its_guides(tmp_path, scan_and_lobes):
    scan, _ = scan_and_lobes
    with pytest.raises(ValueError, match="guides from lobes"):
        make_fissure_mask(stack_job(tmp_path, scan, None), TraceEverySliceModel())


def test_a_config_picks_the_annotator_by_kind(tmp_path):
    job = FissureMaskJob.model_validate(
        {"nifti": tmp_path / "a.nii.gz", "mask_path": tmp_path / "m.nii.gz", "annotator": {"kind": "stack", "size": 3}}
    )
    assert isinstance(job.annotator, StackAnnotator) and job.annotator.size == 3


def test_without_guides_the_stack_annotator_needs_no_lobes_and_its_prompt_does_not_mention_guides(tmp_path, scan_and_lobes):
    scan, _ = scan_and_lobes
    model = TraceEverySliceModel()
    job = stack_job(tmp_path, scan, None, annotator=StackAnnotator(size=3, hint_band_mm=None))
    make_fissure_mask(job, model)
    assert len(model.stacks) == 3 and "GUIDES" not in model.instructions[0]
    assert not (job.work_folder / "fissure_bands").exists()
    assert np.asarray(nib.load(job.mask_path).dataobj).any()


def test_a_prompt_that_does_not_match_the_guides_is_refused():
    with pytest.raises(ValueError, match="explains guides"):
        StackAnnotator(hint_band_mm=None, prompt=DEFAULT_PROMPT)
    with pytest.raises(ValueError, match="does not explain"):
        StackAnnotator(prompt=NO_GUIDES_PROMPT)


def test_the_stack_annotator_can_show_the_agent_the_ridge_map(tmp_path, scan_and_lobes):
    import base64
    import io

    from PIL import Image

    from fissure_annotation.stack_agent import RidgePictures

    scan, _ = scan_and_lobes
    model = TraceEverySliceModel()
    requests = []
    create = model.create
    model.create = lambda **request: requests.append(request) or create(**request)
    annotator = StackAnnotator(size=3, hint_band_mm=None, pictures=RidgePictures(display_max=20))
    job = stack_job(tmp_path, scan, None, annotator=annotator)
    make_fissure_mask(job, model)
    first = next(p for p in requests[0]["input"][0]["content"] if p["type"] == "input_image")
    seen = np.asarray(Image.open(io.BytesIO(base64.b64decode(first["image_url"].split(",")[1]))).convert("L"))
    assert (seen == 0).mean() > 0.5 and seen.max() > 100  # a ridge map is black with a bright line, the CT slice is lung grey
    assert np.asarray(nib.load(job.mask_path).dataobj).any()
