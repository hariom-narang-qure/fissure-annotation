import itertools

import pytest

from fissure_annotation import prompts
from fissure_annotation.config import AgentConfig
from fissure_annotation.prompts import ScrollPromptFacts, scroll_instructions, with_plain_images, with_review_paragraph

FACTS = ScrollPromptFacts(start=150, plane="coronal", top="superior (head end)", lo=40, hi=380, mm_step=0.75)
SETUPS = list(itertools.product((True, False), repeat=3))


def build(tracer=True, reviewer=False, plain=False) -> str:
    return scroll_instructions(AgentConfig(use_tracer=tracer, use_reviewer=reviewer, plain_images=plain), FACTS)


@pytest.mark.parametrize("tracer, reviewer, plain", SETUPS)
def test_no_placeholder_is_left_unfilled(tracer, reviewer, plain):
    text = build(tracer, reviewer, plain)
    for name in ("start", "plane", "top", "lo", "hi", "mm_step", "five_mm", "max_reviews"):
        assert "{" + name + "}" not in text


def test_the_run_facts_are_in_the_text():
    text = build()
    assert "150" in text and "coronal" in text and "superior (head end)" in text


def test_the_no_tracer_prompt_has_no_tracer_vocabulary():
    assert "snap" not in build(tracer=False).lower().replace("snapshot", "")
    assert "snap" in build(tracer=True).lower()


def test_plain_images_drop_the_tick_mark_wording():
    assert "tick marks" in build(plain=False)
    assert "tick marks and labels are drawn" not in build(plain=True)


def test_the_review_paragraph_is_only_present_when_asked_for():
    assert "Review (optional)" in build(reviewer=True)
    assert "Review (optional)" not in build(reviewer=False)


def test_the_review_paragraph_comes_before_the_suggested_way_to_work():
    text = build(reviewer=True)
    assert text.index("Review (optional)") < text.index("Suggested way to work")


def test_max_reviews_is_filled_in():
    text = scroll_instructions(AgentConfig(use_reviewer=True, max_reviews=7), FACTS)
    assert "at most 7 reviews" in text


def test_five_millimetres_is_converted_to_a_slice_step():
    assert "to move about 5 mm you change the slice number by about 7" in build()
    assert "0.8 mm apart" in build()


def test_a_slice_step_wider_than_five_millimetres_still_moves_one_slice():
    wide = FACTS.model_copy(update={"mm_step": 8.0})
    assert "by about 1)" in scroll_instructions(AgentConfig(), wide)


def test_an_edit_whose_text_is_gone_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="plain-image edit"):
        with_plain_images("a template that was reworded")
    with pytest.raises(ValueError, match="review paragraph"):
        with_review_paragraph("a template with no suggested way to work section")


def test_the_edits_are_applied_to_the_template_text_not_a_copy_of_the_file():
    assert prompts.TEMPLATES[True] != with_plain_images(prompts.TEMPLATES[True])
