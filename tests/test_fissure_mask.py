import nibabel as nib
import numpy as np

from fissure_annotation.fissure_mask import FissureMaskJob, make_fissure_mask
from test_agent import FINISH, FakeModel
from test_experiment import ModelPerSlice

LUNG_HU, LINE_HU = -850, -600


def scan_with_a_bright_line(path):
    "512 x 512 x 3 lung with a bright line along row 300 (columns 100 to 400) of the slice as the model sees it"
    volume = np.full((512, 512, 3), LUNG_HU, dtype=np.int16)
    volume[511 - 399:511 - 100 + 1, 511 - 300, :] = LINE_HU  # the slice flips x and y, see slices.slice_view
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)


class CountingModel(ModelPerSlice):
    def __init__(self):
        super().__init__()
        self.conversations = 0

    def create(self, **request):
        self.conversations += "previous_response_id" not in request
        return super().create(**request)


def test_the_mask_has_the_traced_line_on_the_slices_in_range_and_nothing_elsewhere(tmp_path):
    scan_with_a_bright_line(tmp_path / "scan.nii.gz")
    job = FissureMaskJob(nifti=tmp_path / "scan.nii.gz", mask_path=tmp_path / "out" / "scan_fissures.nii.gz", first_slice=1, last_slice=2, workers=2)
    make_fissure_mask(job, ModelPerSlice())
    mask = np.asarray(nib.load(job.mask_path).dataobj)
    assert mask.shape == (512, 512, 3) and mask.dtype == np.uint8
    assert not mask[:, :, 0].any() and mask[511 - 250, 511 - 300, 1] == 1 and mask[511 - 250, 511 - 300, 2] == 1


def test_a_coronal_job_stacks_the_traced_lines_along_y_and_gives_the_model_the_coronal_instructions(tmp_path):
    volume = np.full((512, 4, 300), LUNG_HU, dtype=np.int16)
    volume[511 - 399:511 - 100 + 1, :, 299 - 150] = LINE_HU  # a line along row 150, columns 100 to 400 of every coronal slice
    nib.save(nib.Nifti1Image(volume, np.eye(4)), tmp_path / "scan.nii.gz")
    trace = ("trace_points", {"fissures": [{"name": "oblique", "points": [[100, 150], [399, 150]]}]})
    model = FakeModel([trace], [FINISH], [trace], [FINISH])  # one worker, so the two slices run one after the other
    job = FissureMaskJob(nifti=tmp_path / "scan.nii.gz", mask_path=tmp_path / "coronal.nii.gz", view="coronal", first_slice=1, last_slice=2, workers=1)
    make_fissure_mask(job, model)
    mask = np.asarray(nib.load(job.mask_path).dataobj)
    assert mask.shape == (512, 4, 300) and not mask[:, 0].any() and mask[511 - 250, 1, 299 - 150] == 1 and mask[511 - 250, 2, 299 - 150] == 1
    assert "coronal" in model.requests[0]["instructions"] and "512x300" in str(model.requests[0]["input"])


def test_running_again_only_asks_the_model_about_slices_that_have_no_log(tmp_path):
    scan_with_a_bright_line(tmp_path / "scan.nii.gz")
    job = FissureMaskJob(nifti=tmp_path / "scan.nii.gz", mask_path=tmp_path / "scan_fissures.nii.gz", first_slice=0, last_slice=1, workers=2)
    make_fissure_mask(job, ModelPerSlice())
    (job.work_folder / "run" / "logs" / "slice_0001.json").unlink()
    again = CountingModel()
    make_fissure_mask(job, again)
    assert again.conversations == 1


def test_the_work_folder_defaults_to_a_folder_next_to_the_mask_named_after_the_view(tmp_path):
    job = FissureMaskJob(nifti=tmp_path / "scan.nii.gz", mask_path=tmp_path / "out" / "scan_fissures.nii.gz")
    assert job.work_folder == tmp_path / "out" / "scan_fissures_axial_work"
