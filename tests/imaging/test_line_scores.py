import numpy as np

from fissure_annotation.imaging.line_scores import reference_fissure_lines, score_lines

LEFT_UPPER, LEFT_LOWER = 10, 11


def two_lobes_split_at_row(row: int) -> np.ndarray:
    lobes = np.zeros((100, 100), dtype=np.uint8)
    lobes[10:row, 10:90] = LEFT_UPPER
    lobes[row:90, 10:90] = LEFT_LOWER
    return lobes


def test_the_reference_is_the_two_rows_where_the_lobes_meet():
    lines = reference_fissure_lines(two_lobes_split_at_row(50))
    assert set(np.unique(np.nonzero(lines)[0])) == {49, 50}


def test_a_line_on_the_border_scores_full_precision_and_recall():
    traced = np.zeros((100, 100), dtype=bool)
    traced[50, 10:90] = True
    score = score_lines(traced, two_lobes_split_at_row(50), (1.0, 1.0))
    assert score.traced_near_reference_px == score.traced_px == 80
    assert score.reference_near_traced_px == score.reference_px


def test_a_line_further_than_the_tolerance_scores_zero():
    traced = np.zeros((100, 100), dtype=bool)
    traced[70, 10:90] = True
    score = score_lines(traced, two_lobes_split_at_row(50), (1.0, 1.0), tolerance_mm=3.0)
    assert score.traced_near_reference_px == 0 and score.reference_near_traced_px == 0


def test_the_tolerance_is_in_millimetres_along_each_axis():
    "rows 2 mm apart put a line 2 rows off the border 4 mm away, outside a 3 mm tolerance"
    traced = np.zeros((100, 100), dtype=bool)
    traced[52, 10:90] = True
    lobes = two_lobes_split_at_row(50)
    assert score_lines(traced, lobes, (1.0, 1.0)).traced_near_reference_px == 80
    assert score_lines(traced, lobes, (2.0, 1.0)).traced_near_reference_px == 0


def test_nothing_traced_scores_zero_recall_without_failing():
    score = score_lines(np.zeros((100, 100), dtype=bool), two_lobes_split_at_row(50), (1.0, 1.0))
    assert score.traced_px == 0 and score.reference_near_traced_px == 0
