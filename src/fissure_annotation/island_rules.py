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
- Each rule adds what it removed to the removed voxels as it runs, under its own label.
- A new rule is a new class added to `IslandRule`, and to the config. No other code changes.
"""

from typing import Annotated, Literal

import numpy as np
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, Field

from fissure_annotation.imaging.islands import Islands
from fissure_annotation.removals import RemovedVoxels

# Half the median distance between the centroids of the two largest islands, on 11 scans without severe disease.
# - The median was 164 mm. In 10 scans the two largest islands were on opposite sides, at 140 to 184 mm.
# - In the eleventh (840568) both were on one side, 47 mm apart.
# - The median itself would flag half of the good scans, so the cut sits halfway to it.
MIN_CENTROID_DISTANCE_MM = 82.0


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


class FlagExtraIslands(Rule):
    "more islands than the two main sheets are left, so a person decides which are fissure"

    kind: Literal["flag_extra_islands"] = "flag_extra_islands"
    max_islands: int = Field(default=2, ge=1)

    def apply(self, islands: Islands) -> RuleOutcome:
        if islands.count <= self.max_islands:
            return RuleOutcome(islands)
        flag = ReviewFlag(
            rule=self.kind,
            reason=f"{islands.count} islands are left, more than {self.max_islands}",
        )
        return RuleOutcome(islands, [flag])


class FlagTooFewIslands(Rule):
    "fewer islands than the two main sheets are left, so a lung likely has no fissure"

    kind: Literal["flag_too_few_islands"] = "flag_too_few_islands"
    min_islands: int = Field(default=2, ge=1)

    def apply(self, islands: Islands) -> RuleOutcome:
        if islands.count >= self.min_islands:
            return RuleOutcome(islands)
        flag = ReviewFlag(
            rule=self.kind,
            reason=f"islands left: {islands.count}, fewer than {self.min_islands}",
        )
        return RuleOutcome(islands, [flag])


class FlagCloseMainIslands(Rule):
    """the two largest islands should be the two lungs, so their centroids are far apart

    Two close main islands are likely one lung's fissure in two pieces, with the other lung's fissure missing.
    With fewer than two islands there is nothing to measure, and `FlagTooFewIslands` flags it.
    """

    kind: Literal["flag_close_main_islands"] = "flag_close_main_islands"
    min_centroid_distance_mm: float = Field(default=MIN_CENTROID_DISTANCE_MM, gt=0)

    def apply(self, islands: Islands) -> RuleOutcome:
        if islands.count < 2:
            return RuleOutcome(islands)
        distance = islands.two_largest_centroid_distance_mm()
        if distance >= self.min_centroid_distance_mm:
            return RuleOutcome(islands)
        return RuleOutcome(
            islands,
            [self.flag(f"the two largest islands are {distance:.0f} mm apart, under {self.min_centroid_distance_mm:.0f} mm")],
        )

    def flag(self, reason: str) -> ReviewFlag:
        return ReviewFlag(rule=self.kind, reason=reason)


IslandRule = Annotated[
    DropSmallIslands | FlagExtraIslands | FlagTooFewIslands | FlagCloseMainIslands,
    Field(discriminator="kind"),
]


def default_island_rules() -> list[IslandRule]:
    return [DropSmallIslands(), FlagExtraIslands(), FlagTooFewIslands(), FlagCloseMainIslands()]


class IslandReport(BaseModel):
    "what the rules did to one sheet, saved as part of the post-process report"

    island_voxels_before: list[int]
    island_voxels_after: list[int]
    flags: list[ReviewFlag]

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


class TrimmedSheet:
    def __init__(self, kept: Bool[np.ndarray, "x y z"], report: IslandReport):
        self.kept = kept
        self.report = report


def apply_island_rules(
    rules: list[IslandRule],
    sheet: Bool[np.ndarray, "x y z"],
    affine: Float[np.ndarray, "4 4"],
    removed: RemovedVoxels,
) -> TrimmedSheet:
    before = Islands(sheet, affine)
    islands, flags = before, []
    for rule in rules:
        outcome = rule.apply(islands)
        removed.add(rule.kind, islands.mask & ~outcome.kept.mask)
        islands = outcome.kept
        flags += outcome.flags
    report = IslandReport(
        island_voxels_before=before.sizes.tolist(),
        island_voxels_after=islands.sizes.tolist(),
        flags=flags,
    )
    return TrimmedSheet(islands.mask, report)
