import json

import numpy as np

from fissure_annotation.annotators import StackAnnotator
from fissure_annotation.eval_set import EvalCase, cases_for
from fissure_annotation.imaging.png import save_png
from fissure_annotation.slice_eval import SliceEval, SliceEvalRun, plan_lines, run_slice_eval, score_rows, stack_around, summary_lines
from test_fissure_mask import StackLineModel, scan_with_a_bright_line

PRICES = {"default": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.0, "output": 5.0}}


def test_a_ground_truth_slice_sits_in_the_middle_of_its_stack():
    assert stack_around(200, step=3, size=5, slice_count=400) == [194, 197, 200, 203, 206]


def test_a_stack_near_the_edge_of_the_scan_shifts_inwards():
    assert stack_around(2, step=3, size=5, slice_count=400) == [2, 5, 8, 11, 14]
    assert stack_around(398, step=3, size=5, slice_count=400) == [386, 389, 392, 395, 398]


def write_case(root, name, tests, gt):
    "gt maps an axial slice index to the row of its reference line, or None for a slice with no fissure"
    folder = root / name
    folder.mkdir(parents=True)
    scan_with_a_bright_line(folder / "scan.nii.gz", slices=7)
    case_meta = {"tests": tests, "slices": [{"view": "axial", "index": z} for z in gt], "source": "test"}
    (folder / "meta.json").write_text(json.dumps(case_meta))
    for z, row in gt.items():
        mask = np.zeros((512, 512), dtype=np.uint8)
        if row is not None:
            mask[row, 100:400] = 255
        save_png(mask, EvalCase(folder).gt_path("axial", z))
    return folder


def test_the_harness_gathers_the_cases_whose_meta_lists_the_test(tmp_path):
    write_case(tmp_path, "b", ["slice"], {0: 300})
    write_case(tmp_path, "a", ["slice"], {0: 300})
    write_case(tmp_path, "scan_only", [], {0: 300})
    assert [case.name for case in cases_for(tmp_path, "slice")] == ["a", "b"]


def settings_for(tmp_path, **overrides) -> SliceEval:
    return SliceEval(
        **{
            "eval_root": tmp_path / "eval",
            "out_dir": tmp_path / "out",
            "annotator": StackAnnotator(size=3),
            "slice_step": 3,
            "repeats": 1,
            "workers": 1,
            "prices": PRICES,
            **overrides,
        }
    )


def test_a_trace_on_the_reference_scores_full_and_an_empty_reference_counts_no_reference(tmp_path):
    "the model traces row 300 on every slice; slice 6 has no fissure in its ground truth"
    write_case(tmp_path / "eval", "case", ["slice"], {0: 300, 3: 300, 6: None})
    settings = settings_for(tmp_path)
    model = StackLineModel()
    run = run_slice_eval(settings, model)
    assert sorted(model.stacks) == [[0, 3, 6], [0, 3, 6], [0, 3, 6]]  # one conversation per ground truth slice, shifted inwards
    rows = sorted(score_rows(run), key=lambda row: row.slice_index)
    assert [(r.slice_index, r.traced_near_reference_px == r.traced_px > 0) for r in rows] == [(0, True), (3, True), (6, False)]
    assert rows[2].reference_px == 0 and rows[2].traced_px > 0  # a trace where there is no fissure lowers precision
    assert (settings.out_dir / "case" / "axial" / "slice_0006" / "repeat0" / "stack_logs").exists()
    summary = summary_lines(run, rows)
    assert summary[0].startswith("repeat 0: 3 slices, precision 66.7% recall 100.0%")


def test_a_finished_stack_is_not_run_again_and_max_stacks_limits_the_eval(tmp_path):
    write_case(tmp_path / "eval", "case", ["slice"], {0: 300, 6: 300})
    settings = settings_for(tmp_path, max_stacks=1)
    first = StackLineModel()
    run_slice_eval(settings, first)
    again = StackLineModel()
    run_slice_eval(settings, again)
    assert len(first.stacks) == 1 and again.stacks == []
    assert "1 stacks x 1 repeats, one scored slice each" in plan_lines(SliceEvalRun(settings))[-1]
