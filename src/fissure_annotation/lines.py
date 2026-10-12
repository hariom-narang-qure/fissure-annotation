"""The data the agents pass around: the lines they draw, what they submit, and the fissures that result.

- A `Line` is stored in the main frame (see `frames`), so it is independent of the view it was drawn in.
- A `SubmittedLine` is the model's verdict on one of its lines, a `Fissure` is the deliverable made from the two.
"""
from typing import Literal

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

FissureName = Literal["oblique", "horizontal", "accessory", "unknown"]
Side = Literal["right", "left"]
FissureSide = Literal["right", "left", "unknown"]
"""A fissure the scan agent files as accessory has no side it was told to trust."""
Geometry = Literal["mine", "traced"]
Status = Literal["confirmed", "uncertain", "dropped"]
Confidence = Literal["clear", "faint", "uncertain"]

CONFIDENCE_OF_STATUS: dict[str, Confidence] = {"confirmed": "clear", "uncertain": "faint"}

FitStats = dict[str, float | list[float]]
"""What `tracing.snap` reports, empty when the line was not traced."""

TracedPath = Float[np.ndarray, "n 2"]
"""(row, col) in slice pixels, one point per pixel step."""


class MainPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class Waypoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int | float
    y: int | float
    """Whole numbers from the agents. A one-shot reply may hold decimals, which are kept as given."""


class Line(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: FissureName
    side: Side
    own_points: list[MainPoint]
    """The model's own waypoints, in the main frame."""
    traced_path: TracedPath | None = None
    """None when no tracer ran, the line is then the own points joined by straight segments."""
    fit: FitStats = {}

    @property
    def is_traced(self) -> bool:
        return self.traced_path is not None


class SubmittedLine(BaseModel):
    line_id: int
    status: Status
    geometry: Geometry = "mine"
    evidence: str


class DrawableFissure(BaseModel):
    "what drawing and scoring need from any fissure, whichever pipeline made it"

    name: FissureName
    side: FissureSide
    confidence: Confidence
    waypoints: list[Waypoint]


class ProposedFissure(DrawableFissure):
    "one fissure from a one-shot reply, before any tool has checked it"

    evidence: str = ""


class Fissure(DrawableFissure):
    "the deliverable of an agent run"

    evidence: str
    geometry: Geometry
    fit: FitStats
