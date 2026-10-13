"""What a model decides about one slice, and where a run keeps its files.

The waypoints are saved before anything is traced, so the manual part of every run can be reloaded and traced again with a different tracer.
The JSON layout is the one earlier runs wrote, so their points load too.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from fissure_annotation.imaging.tracing import Point


class Fissure(BaseModel):
    name: str
    points: list[Point] = Field(min_length=2)


class SliceWaypoints(BaseModel):
    slice_index: int
    fissures: list[Fissure]
    notes: str = ""


class RunFolder(BaseModel):
    "one folder per run: points/, masks/, overlays/ and logs/, each file named after its slice PNG"

    root: Path

    def points_path(self, slice_png: Path) -> Path:
        return self.root / "points" / f"{slice_png.stem}.json"

    def mask_path(self, slice_png: Path) -> Path:
        return self.root / "masks" / f"{slice_png.stem}_mask.png"

    def overlay_path(self, slice_png: Path) -> Path:
        return self.root / "overlays" / f"{slice_png.stem}_overlay.png"

    def log_path(self, slice_png: Path) -> Path:
        return self.root / "logs" / f"{slice_png.stem}.json"

    def save_waypoints(self, slice_png: Path, waypoints: SliceWaypoints) -> None:
        path = self.points_path(slice_png)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(waypoints.model_dump_json(indent=1))

    def load_waypoints(self, slice_png: Path) -> SliceWaypoints:
        return SliceWaypoints.model_validate_json(
            self.points_path(slice_png).read_text()
        )
