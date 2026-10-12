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
Geometry = Literal["mine", "traced"]
Status = Literal["confirmed", "uncertain", "dropped"]
Confidence = Literal["clear", "faint"]

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

    x: int
    y: int


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


class Fissure(BaseModel):
    name: FissureName
    side: Side
    confidence: Confidence
    waypoints: list[Waypoint]
    evidence: str
    geometry: Geometry
    fit: FitStats
