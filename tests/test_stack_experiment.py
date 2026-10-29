import nibabel as nib
import numpy as np
import pytest

from fissure_annotation.imaging.slices import slice_view, write_slices
from fissure_annotation.stack_experiment import Stack, StackExperiment, all_score_rows, run_stack_experiment
from test_agent import FakeModel

PRICES = {"default": {"input": 2.0, "cached_input": 0.1, "cache_write": 2.5, "output": 10.0}}


@pytest.fixture
def run_dir(tmp_path):
    "a scan run of 16 axial slices whose upper and lower lobes meet at row 30 of every slice"
    run_dir = tmp_path / "scan" / "flex_step4"
    run_dir.mkdir(parents=True)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.full((64, 64, 16), -800, dtype=np.int16), affine), run_dir / "scan.nii.gz")
    lobes = np.zeros((64, 64, 16), dtype=np.uint8)
    lobes[8:56, 8:56, :] = 10
    lobes[8:56, 8:34, :] = 11  # rows of an axial slice run along y reversed, see slices.slice_view
    nib.save(nib.Nifti1Image(lobes, affine), run_dir / "lobes.nii.gz")
    write_slices(run_dir / "scan.nii.gz", run_dir / "fissures_work" / "axial" / "slices", "axial")
    return run_dir


def border_row(run_dir) -> int:
    lobes = slice_view(np.asarray(nib.load(run_dir / "lobes.nii.gz").dataobj), "axial", 0)
    return int(np.argmax((lobes[:, 30] == 10) & (np.roll(lobes[:, 30], -1) == 11)))


def experiment(tmp_path, run_dir, sizes):
    return StackExperiment(
        out_dir=tmp_path / "out",
        stacks=[Stack(run_dir=run_dir, view="axial", centre=8, hint_lobes=run_dir / "lobes.nii.gz")],
        stack_sizes=sizes,
        slice_step=4,
        prices=PRICES,
    )


def test_a_stack_traced_on_the_border_scores_full_precision_and_the_empty_baseline_none(tmp_path, run_dir):
    row = border_row(run_dir)
    line = lambda z: {"slice": z, "name": "oblique", "points": [[10, row], [53, row]]}
    model = FakeModel([("trace_points", {"fissures": [line(4), line(8), line(12)]})], [("finish", {"summary": "ok"})])
    job = experiment(tmp_path, run_dir, [3])
    run_stack_experiment(job, model)
    rows = all_score_rows(job)
    ours = [r for r in rows if r.variant == "size3"]
    assert [r.offset for r in ours] == [-1, 0, 1]
    assert all(r.traced_px > 0 and r.traced_near_reference_px == r.traced_px for r in ours)
    assert all(r.traced_px == 0 for r in rows if r.variant == "baseline")


def test_a_finished_conversation_is_not_run_again(tmp_path, run_dir):
    job = experiment(tmp_path, run_dir, [1])
    run_stack_experiment(job, FakeModel([("finish", {"summary": "none"})]))
    second = FakeModel()
    run_stack_experiment(job, second)
    assert second.requests == []


def test_even_stack_sizes_are_refused(tmp_path, run_dir):
    with pytest.raises(ValueError):
        experiment(tmp_path, run_dir, [2])


def test_hint_lobes_give_band_pngs_on_the_slice_grid(tmp_path, run_dir):
    from fissure_annotation.imaging.png import read_gray

    job = StackExperiment(
        out_dir=tmp_path / "out",
        stacks=[Stack(run_dir=run_dir, view="axial", centre=8, hint_lobes=run_dir / "lobes.nii.gz")],
        stack_sizes=[1],
        prices=PRICES,
    )
    model = FakeModel([("finish", {"summary": "none"})])
    run_stack_experiment(job, model)
    hints = read_gray(job.bands_dir(job.stacks[0]) / "slice_0008.png")
    ct = read_gray(run_dir / "fissures_work" / "axial" / "slices" / "slice_0008.png")
    assert hints.shape == ct.shape
    assert set(np.unique(hints)) == {0, 1}  # the test lobes are the left upper and lower, so only the left oblique band
    assert abs(int(np.nonzero(hints[:, 30])[0].mean()) - border_row(run_dir)) <= 1
