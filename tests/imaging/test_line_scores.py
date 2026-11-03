import numpy as np

from fissure_annotation.imaging.line_scores import score_lines


def line_at_row(row: int, first_column: int = 10, last_column: int = 90) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[row, first_column:last_column] = True
    return mask


def test_a_line_on_the_reference_scores_full_precision_and_recall():
    score = score_lines(line_at_row(50), line_at_row(50), (1.0, 1.0))
    assert score.traced_near_reference_px == score.traced_px == 80
    assert score.reference_near_traced_px == score.reference_px == 80


def test_a_line_further_than_the_tolerance_scores_zero():
    score = score_lines(line_at_row(70), line_at_row(50), (1.0, 1.0), tolerance_mm=3.0)
    assert score.traced_near_reference_px == 0 and score.reference_near_traced_px == 0


def test_the_tolerance_is_in_millimetres_along_each_axis():
    "rows 2 mm apart put a line 2 rows off the reference 4 mm away, outside a 3 mm tolerance"
    assert score_lines(line_at_row(52), line_at_row(50), (1.0, 1.0)).traced_near_reference_px == 80
    assert score_lines(line_at_row(52), line_at_row(50), (2.0, 1.0)).traced_near_reference_px == 0


def test_a_short_trace_has_full_precision_and_partial_recall():
    score = score_lines(line_at_row(50, 10, 50), line_at_row(50), (1.0, 1.0), tolerance_mm=0.5)
    assert score.traced_near_reference_px == score.traced_px == 40
    assert score.reference_near_traced_px == 40 and score.reference_px == 80


def test_nothing_traced_or_no_reference_scores_zero_without_failing():
    empty = np.zeros((100, 100), dtype=bool)
    assert score_lines(empty, line_at_row(50), (1.0, 1.0)).reference_near_traced_px == 0
    assert score_lines(line_at_row(50), empty, (1.0, 1.0)).traced_near_reference_px == 0
