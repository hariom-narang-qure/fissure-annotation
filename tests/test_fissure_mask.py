import re
import threading

import nibabel as nib
import numpy as np
import pytest

from fissure_annotation.annotators import StackAnnotator
from fissure_annotation.fissure_mask import (
    FissureMaskJob,
    make_fissure_mask,
    view_mask_path,
)
from test_agent import FINISH, FakeModel

LUNG_HU, LINE_HU = -850, -600


def scan_with_a_bright_line(path, slices=3):
    "512 x 512 lung with a bright line along row 300 (columns 100 to 400) of each slice as the model sees it"
    volume = np.full((512, 512, slices), LUNG_HU, dtype=np.int16)
    volume[511 - 399 : 511 - 100 + 1, 511 - 300, :] = (
        LINE_HU  # the slice flips x and y, see slices.slice_view
    )
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)


class StackLineModel:
    """each conversation traces the same line on every slice of its first message, then finishes

    Stacks run in threads, so each thread keeps its own scripted model.
    """

    def __init__(self, points=((100, 300), (399, 300))):
        self.points = [list(point) for point in points]
        self.local, self.responses = threading.local(), self
        self.stacks, self.instructions = [], []

    @property
    def conversations(self) -> int:
        return len(self.stacks)

    def create(self, **request):
        if "previous_response_id" not in request:
            labels = [part["text"] for part in request["input"][0]["content"] if part["type"] == "input_text"]
            stack = [int(re.match(r"Slice (\d+):", label).group(1)) for label in labels[1:]]
            self.stacks.append(stack)
            self.instructions.append((request["instructions"], str(request["input"])))
            fissures = [{"slice": z, "name": "line", "points": self.points} for z in stack]
            self.local.model = FakeModel([("trace_points", {"fissures": fissures})], [FINISH])
        return self.local.model.create(**request)


ONE_SLICE_STACKS = StackAnnotator(size=1)


def test_the_mask_has_the_traced_line_on_the_slices_in_range_and_nothing_elsewhere(
    tmp_path,
):
    scan_with_a_bright_line(tmp_path / "scan.nii.gz")
    job = FissureMaskJob(
        nifti=tmp_path / "scan.nii.gz",
        mask_path=tmp_path / "out" / "scan_fissures.nii.gz",
        first_slice=1,
        last_slice=2,
        workers=2,
    )
    make_fissure_mask(job, StackLineModel())
    mask = np.asarray(nib.load(job.mask_path).dataobj)
    assert mask.shape == (512, 512, 3) and mask.dtype == np.uint8
    assert (
        not mask[:, :, 0].any()
        and mask[511 - 250, 511 - 300, 1] == 1
        and mask[511 - 250, 511 - 300, 2] == 1
    )


def test_a_coronal_job_stacks_the_traced_lines_along_y_and_gives_the_model_the_coronal_instructions(
    tmp_path,
):
    volume = np.full((512, 4, 300), LUNG_HU, dtype=np.int16)
    volume[511 - 399 : 511 - 100 + 1, :, 299 - 150] = (
        LINE_HU  # a line along row 150, columns 100 to 400 of every coronal slice
    )
    nib.save(nib.Nifti1Image(volume, np.eye(4)), tmp_path / "scan.nii.gz")
    model = StackLineModel([[100, 150], [399, 150]])
    job = FissureMaskJob(
        nifti=tmp_path / "scan.nii.gz",
        mask_path=tmp_path / "coronal.nii.gz",
        views=["coronal"],
        first_slice=1,
        last_slice=2,
        workers=1,
    )
    make_fissure_mask(job, model)
    mask = np.asarray(nib.load(job.mask_path).dataobj)
    assert (
        mask.shape == (512, 4, 300)
        and not mask[:, 0].any()
        and mask[511 - 250, 1, 299 - 150] == 1
        and mask[511 - 250, 2, 299 - 150] == 1
    )
    instructions, first_message = model.instructions[0]
    assert "coronal" in instructions and "512x300" in first_message


def test_running_again_only_asks_the_model_about_slices_that_have_no_log(tmp_path):
    scan_with_a_bright_line(tmp_path / "scan.nii.gz")
    job = FissureMaskJob(
        nifti=tmp_path / "scan.nii.gz",
        mask_path=tmp_path / "scan_fissures.nii.gz",
        first_slice=0,
        last_slice=1,
        workers=2,
        annotator=ONE_SLICE_STACKS,
    )
    make_fissure_mask(job, StackLineModel())
    (job.work_folder / "axial" / "run" / "stack_logs" / "stack_0001_0001.json").unlink()
    again = StackLineModel()
    make_fissure_mask(job, again)
    assert again.stacks == [[1]]


def test_the_work_folder_defaults_to_a_folder_next_to_the_mask(tmp_path):
    job = FissureMaskJob(
        nifti=tmp_path / "scan.nii.gz",
        mask_path=tmp_path / "out" / "scan_fissures.nii.gz",
    )
    assert job.work_folder == tmp_path / "out" / "scan_fissures_work"


def line_on_every_slice_model() -> StackLineModel:
    "row 2, columns 2 to 15, which exists in both the axial (20 x 6) and the coronal (20 x 5) slices of the small scan below"
    return StackLineModel([[2, 2], [15, 2]])


def test_the_final_mask_is_the_or_of_the_masks_of_every_view(tmp_path):
    nib.save(
        nib.Nifti1Image(np.full((20, 6, 5), LUNG_HU, dtype=np.int16), np.eye(4)),
        tmp_path / "scan.nii.gz",
    )

    def mask_of(views):
        job = FissureMaskJob(
            nifti=tmp_path / "scan.nii.gz",
            mask_path=tmp_path / f"{'_'.join(views)}.nii.gz",
            views=views,
            workers=1,
        )
        make_fissure_mask(job, line_on_every_slice_model())
        return np.asarray(nib.load(job.mask_path).dataobj)

    axial, coronal, both = (
        mask_of(["axial"]),
        mask_of(["coronal"]),
        mask_of(["axial", "coronal"]),
    )
    assert (
        axial[10, 3, 0] == 1 and axial[10, 0, 2] == 0
    )  # axial traces y=3 on every z, not z=2 on every y
    assert coronal[10, 0, 2] == 1 and coronal[10, 3, 0] == 0
    assert (
        (both == (axial | coronal)).all()
        and both.max() == 1
        and both.sum() < axial.sum() + coronal.sum()
    )  # they overlap at y=3, z=2


def test_a_slice_range_needs_a_single_view(tmp_path):
    with pytest.raises(ValueError, match="exactly one view"):
        FissureMaskJob(
            nifti=tmp_path / "a.nii.gz",
            mask_path=tmp_path / "m.nii.gz",
            views=["axial", "coronal"],
            first_slice=5,
        )


def test_the_mask_of_each_view_is_saved_next_to_the_combined_mask(tmp_path):
    nib.save(
        nib.Nifti1Image(np.full((20, 6, 5), LUNG_HU, dtype=np.int16), np.eye(4)),
        tmp_path / "scan.nii.gz",
    )
    job = FissureMaskJob(
        nifti=tmp_path / "scan.nii.gz",
        mask_path=tmp_path / "both.nii.gz",
        views=["axial", "coronal"],
        workers=1,
    )
    make_fissure_mask(job, line_on_every_slice_model())

    def load(path):
        return np.asarray(nib.load(path).dataobj)

    axial_path = view_mask_path(job.mask_path, "axial")
    axial, coronal = (
        load(axial_path),
        load(view_mask_path(job.mask_path, "coronal")),
    )
    assert axial_path == tmp_path / "both_axial.nii.gz"
    assert axial[10, 3, 0] == 1 and axial[10, 0, 2] == 0  # only the axial marks
    assert coronal[10, 0, 2] == 1 and coronal[10, 3, 0] == 0  # only the coronal marks
    assert (load(job.mask_path) == (axial | coronal)).all()


def test_a_slice_step_sends_only_every_step_th_slice_and_the_last_one_to_the_model(
    tmp_path,
):
    volume = np.full((512, 512, 5), LUNG_HU, dtype=np.int16)
    volume[511 - 399 : 511 - 100 + 1, 511 - 300, :] = LINE_HU
    nib.save(nib.Nifti1Image(volume, np.eye(4)), tmp_path / "scan.nii.gz")
    model = StackLineModel()
    job = FissureMaskJob(
        nifti=tmp_path / "scan.nii.gz",
        mask_path=tmp_path / "scan_fissures.nii.gz",
        first_slice=0,
        last_slice=4,
        slice_step=3,
        workers=2,
        annotator=ONE_SLICE_STACKS,
    )
    make_fissure_mask(job, model)
    mask = np.asarray(nib.load(job.mask_path).dataobj)
    marked_slices = [z for z in range(5) if mask[:, :, z].any()]
    assert marked_slices == [0, 3, 4]  # 0 and 3 are on the step, 4 is the last slice
    assert model.conversations == 3
