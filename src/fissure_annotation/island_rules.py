"""Rules that trim the islands of a fissure sheet, or flag the scan for a person to review.

Why:
- In a scan without much destruction, the fissures form two large connected sheets.
  - The left oblique fissure is one.
  - The right horizontal fissure meets the right oblique fissure, so the right side is the other.
- Other islands are mostly throwaway: short traces that are not connected to either sheet.
- Measured on 11 scans without severe disease, on the sheet before the 150 voxel filter:
  - In 7 of them the third island was under 10% of the largest, and in 5 under 2%.
  - In one (125572) the third island was 67% of the largest, so a real fissure can be in more than two pieces.
  - That is why a rule may flag a scan instead of trimming it.

How:
- Each rule is one small class with an `apply(islands)` method.
  - A trimming rule returns the islands it keeps.
  - A flagging rule returns the islands unchanged, plus flags.
- The rules run in the order of the config, each on what the previous one kept.
- Flags never change the sheet. They name a scan a person should look at, and why.
- A new rule is a new class added to `IslandRule`, and to the config. No other code changes.
"""

from typing import Annotated, Literal

import numpy as np
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, Field

from fissure_annotation.imaging.islands import Islands


class ReviewFlag(BaseModel):
    rule: str
    reason: str


class RuleOutcome:
    def __init__(self, kept: Islands, flags: list[ReviewFlag] | None = None):
        self.kept = kept
        self.flags = flags or []


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DropSmallIslands(Rule):
    "an island under a share of the largest island is not a fissure"

    kind: Literal["drop_small_islands"] = "drop_small_islands"
    min_share_of_largest: float = Field(default=0.05, ge=0, le=1)

    def apply(self, islands: Islands) -> RuleOutcome:
        if islands.count == 0:
            return RuleOutcome(islands)
        too_small = islands.sizes < self.min_share_of_largest * islands.sizes[0]
        return RuleOutcome(islands.without(np.flatnonzero(too_small)))


IslandRule = Annotated[DropSmallIslands, Field(discriminator="kind")]


def default_island_rules() -> list[IslandRule]:
    return [DropSmallIslands()]


class IslandReport(BaseModel):
    "what the rules did to one sheet, saved next to it as JSON"

    island_voxels_before: list[int]
    island_voxels_after: list[int]
    flags: list[ReviewFlag]

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


class TrimmedSheet:
    def __init__(self, kept: Bool[np.ndarray, "x y z"], removed: Bool[np.ndarray, "x y z"], report: IslandReport):
        self.kept = kept
        self.removed = removed
        self.report = report


def apply_island_rules(
    rules: list[IslandRule],
    sheet: Bool[np.ndarray, "x y z"],
    affine: Float[np.ndarray, "4 4"],
) -> TrimmedSheet:
    before = Islands(sheet, affine)
    islands, flags = before, []
    for rule in rules:
        outcome = rule.apply(islands)
        islands = outcome.kept
        flags += outcome.flags
    report = IslandReport(
        island_voxels_before=before.sizes.tolist(),
        island_voxels_after=islands.sizes.tolist(),
        flags=flags,
    )
    return TrimmedSheet(islands.mask, sheet & ~islands.mask, report)
