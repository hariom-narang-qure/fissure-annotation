import pytest

from fissure_annotation.agent import FULL, Slice
from fissure_annotation.scroll_agent import Stack
from fissure_annotation.slicing import cut


@pytest.fixture
def stack(case) -> Stack:
    return Stack(case, "coronal", use_tracer=False)


def test_lung_range_is_where_enough_lung_exists(stack):
    assert (stack.lo, stack.hi) == (10, 49)


@pytest.mark.parametrize("bad", ["x", None, 3.5e99])
def test_validate_rejects_non_integers_with_the_valid_range(stack, bad):
    index, message = stack.validate(bad)
    assert index is None and "10-49" in message


def test_validate_names_the_nearest_valid_slice_when_out_of_range(stack):
    index, message = stack.validate(70)
    assert index is None and "nearest valid slice is 49" in message


def test_validate_accepts_numeric_strings_inside_the_range(stack):
    assert stack.validate("20") == (20, "")


def test_all_slices_share_one_crop_box_and_line_ids(stack):
    a, b = stack.get(15), stack.get(40)
    assert a.prepared.box == b.prepared.box == stack.box
    assert a.next_id() == 1 and b.next_id() == 2


def test_frame_round_trip_between_model_and_slice_pixels(case):
    sl = Slice(cut(case, "coronal", 30), use_tracer=False)
    for x, y in [(0, 0), (500, 500), (1000, 1000), (123, 876)]:
        col, row = sl.to_px(x, y)
        assert sl.to_model(col, row) == (round(x), round(y))


def test_view_frame_round_trip_inside_a_zoom(case):
    sl = Slice(cut(case, "coronal", 30), use_tracer=False)
    box = (200.0, 300.0, 600.0, 700.0)
    main = Slice.local_to_main(250, 750, box)
    assert main == (300.0, 600.0)
    assert Slice.main_to_local(*main, box) == (250.0, 750.0)


def test_unknown_view_id_is_a_clear_error(case):
    sl = Slice(cut(case, "coronal", 30), use_tracer=False)
    with pytest.raises(KeyError, match="unknown view_id"):
        sl.box("v9")
    assert sl.box(FULL) == (0.0, 0.0, 1000.0, 1000.0)
