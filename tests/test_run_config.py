import pytest
from pydantic import ValidationError

from fissure_annotation.config import RunConfig


def test_defaults_match_the_round_13_setup():
    cfg = RunConfig(plane="coronal", start=10)
    assert (cfg.use_tracer, cfg.plain_images, cfg.use_reviewer, cfg.max_steps) == (True, False, False, 24)


def test_unknown_plane_is_rejected():
    with pytest.raises(ValidationError):
        RunConfig(plane="sagittal", start=0)


def test_a_typo_in_a_setting_is_an_error_not_ignored():
    with pytest.raises(ValidationError):
        RunConfig(plane="axial", start=0, use_tracr=False)


def test_the_api_reasoning_dict_comes_from_the_effort():
    assert RunConfig(plane="axial", start=0, reasoning_effort="high").reasoning == {"effort": "high"}


def test_copy_with_a_new_slice_keeps_every_other_setting():
    base = RunConfig(plane="coronal", start=0, use_tracer=False, plain_images=True, max_reviews=5)
    one = base.model_copy(update={"plane": "axial", "start": 7})
    assert (one.plane, one.start) == ("axial", 7)
    assert (one.use_tracer, one.plain_images, one.max_reviews) == (False, True, 5)


def test_configs_are_immutable_so_parallel_runs_cannot_share_state():
    cfg = RunConfig(plane="axial", start=0)
    with pytest.raises(ValidationError):
        cfg.start = 1
