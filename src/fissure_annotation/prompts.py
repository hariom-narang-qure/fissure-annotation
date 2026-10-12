"""Assemble the instruction text of the scrolling agent from the template files in `prompts/`.

- There are two templates, with and without the ridge tracer, so that the no-tracer prompt has no tracer vocabulary at all.
- The plain-image and reviewer variants are small edits of a template, each applied only if the text it edits is still there.
  - That keeps a reworded template from silently dropping an edit.
- The edited template is then filled with the facts of the run.
"""
from pathlib import Path

from pydantic import BaseModel

from fissure_annotation.config import AgentConfig

PROMPT_DIR = Path(__file__).parents[2] / "prompts"
TEMPLATES = {True: (PROMPT_DIR / "fissure_scroll_agent.txt").read_text(),
             False: (PROMPT_DIR / "fissure_scroll_agent_notracer.txt").read_text()}   # keyed by AgentConfig.use_tracer
REVIEW_INSERTED_BEFORE = "\n\nSuggested way to work"
FIVE_MM = 5


PLAIN_PROMPT_EDITS = [("; tick marks and labels are drawn inside the edges of zoomed images every 100 units", "; the images have no tick marks or labels, so place points by eye as a fraction of the image width and height (the centre is 500, 500)"),
                      (" clean, with tick marks, in the view", " clean, in the view")]


REVIEW_PARAGRAPH = ("\n\nReview (optional): `overlay(..., review: true)` also asks a separate, independent reviewer to check your lines against a rule list (ends, no interpolation, points on the line, smooth path, no duplicates, right structure, naming, missed fissures) and to tell you which lines, and which numbered points of them, look problematic and why, in words only (no coordinates). "
                    "Use it once when your lines for the slice are drawn, before you submit; you have at most {max_reviews} reviews per run. The reviewer can be wrong: verify each finding by zooming on that spot before you change anything, and keep a line you can see is right.")


class ScrollPromptFacts(BaseModel):
    "what the template needs to know about this run"

    start: int
    plane: str
    top: str
    lo: int
    hi: int
    mm_step: float


def with_plain_images(template: str) -> str:
    for old, new in PLAIN_PROMPT_EDITS:
        if old not in template:
            raise ValueError(f"the plain-image edit does not apply, this text is not in the template: {old!r}")
        template = template.replace(old, new)
    return template


def with_review_paragraph(template: str) -> str:
    if REVIEW_INSERTED_BEFORE not in template:
        raise ValueError(f"the review paragraph has nowhere to go, this text is not in the template: {REVIEW_INSERTED_BEFORE!r}")
    return template.replace(REVIEW_INSERTED_BEFORE, REVIEW_PARAGRAPH + REVIEW_INSERTED_BEFORE, 1)


def scroll_instructions(agent: AgentConfig, facts: ScrollPromptFacts) -> str:
    template = TEMPLATES[agent.use_tracer]
    if agent.plain_images:
        template = with_plain_images(template)
    if agent.use_reviewer:
        template = with_review_paragraph(template)
    return template.format(max_reviews=agent.max_reviews, five_mm=max(round(FIVE_MM / facts.mm_step), 1), **facts.model_dump())
