"""Settings for one agent run, in one place instead of threaded through function arguments."""
from dataclasses import dataclass, field

MODEL = "@openai/gpt-6.1-sol"


@dataclass(frozen=True)
class RunConfig:
    plane: str                                   # "coronal" | "axial"
    start: int                                   # the slice whose lines are delivered
    model: str = MODEL
    max_steps: int = 24
    max_output_tokens: int = 24000
    reasoning: dict = field(default_factory=lambda: {"effort": "medium"})
    one_call_per_turn: bool = False              # True: only the first tool call of a turn is run (and the API is asked for no parallel calls); round 9 showed it changes nothing
    use_reviewer: bool = False                   # overlay(review=true) asks a separate reviewer agent (reviewer_model) to check the lines against prompts/fissure_reviewer_rules.txt
    max_reviews: int = 2                         # reviews per run
    reviewer_model: str = MODEL
    plain_images: bool = False                   # True: plain PNGs: no tick marks or labels drawn on any image, native resolution (the first image is the native crop, zoomed views are scaled to that same size)
    use_tracer: bool = True                      # False: the model only places waypoints; lines are its points joined by straight segments (no ridge tracer, no fit numbers, no "traced" geometry)

    def __post_init__(self):
        if self.plane not in ("coronal", "axial"):
            raise ValueError(f"unknown plane {self.plane!r}")
