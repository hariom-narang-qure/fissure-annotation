"""Settings for one agent run, in one place instead of threaded through function arguments."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MODEL = "@openai/gpt-6.1-sol"
EXPERIMENT_OUT = Path("work/exp")

Plane = Literal["coronal", "axial"]
ReasoningEffort = Literal["low", "medium", "high"]


class Section(BaseModel):
    "A typo in a config file should fail, not be ignored."

    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentConfig(Section):
    "what distinguishes one agent setup from another, independent of which slice it runs on"

    model: str = MODEL
    max_steps: int = 24
    max_output_tokens: int = 24000
    reasoning_effort: ReasoningEffort = "medium"
    one_call_per_turn: bool = Field(False, description="only the first tool call of a turn is run, and the API is asked for no parallel calls. Round 9 showed it changes nothing")
    use_reviewer: bool = Field(False, description="overlay(review=true) asks a separate reviewer agent to check the lines against prompts/fissure_reviewer_rules.txt")
    max_reviews: int = Field(2, description="reviews per run")
    reviewer_model: str = MODEL
    plain_images: bool = Field(False, description="no tick marks or labels on any image, native resolution: the first image is the native crop and zoomed views are scaled to that size")
    use_tracer: bool = Field(True, description="False: the model only places waypoints and lines are its points joined by straight segments, with no ridge tracer, no fit numbers and no traced geometry")

    @property
    def reasoning(self) -> dict:
        return {"effort": self.reasoning_effort}


class RunConfig(AgentConfig):
    "one agent run: an agent setup on one slice"

    plane: Plane
    start: int = Field(description="the slice whose lines are delivered")
