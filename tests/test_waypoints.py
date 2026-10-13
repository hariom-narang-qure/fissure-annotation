import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fissure_annotation.waypoints import Fissure, RunFolder, SliceWaypoints

SLICE = Path("slices/slice_0160.png")


def test_saved_waypoints_load_back_unchanged(tmp_path):
    folder = RunFolder(root=tmp_path)
    waypoints = SliceWaypoints(
        slice_index=160,
        fissures=[Fissure(name="oblique", points=[(10, 20), (30, 25)])],
        notes="clear",
    )
    folder.save_waypoints(SLICE, waypoints)
    assert folder.load_waypoints(SLICE) == waypoints


def test_points_written_by_the_earlier_runs_still_load(tmp_path):
    folder = RunFolder(root=tmp_path)
    folder.points_path(SLICE).parent.mkdir(parents=True)
    folder.points_path(SLICE).write_text(
        json.dumps(
            {
                "slice_png": "/somewhere/slice_0160.png",
                "slice_index": 160,
                "fissures": [
                    {
                        "name": "image-left lung oblique",
                        "points": [[95, 268], [135, 293]],
                    }
                ],
                "notes": "faint",
            }
        )
    )
    assert folder.load_waypoints(SLICE).fissures[0].points == [(95, 268), (135, 293)]


def test_a_fissure_needs_two_points():
    with pytest.raises(ValidationError):
        Fissure(name="oblique", points=[(10, 20)])


def test_every_artifact_is_named_after_its_slice(tmp_path):
    folder = RunFolder(root=tmp_path)
    assert folder.points_path(SLICE) == tmp_path / "points" / "slice_0160.json"
    assert folder.mask_path(SLICE) == tmp_path / "masks" / "slice_0160_mask.png"
    assert (
        folder.overlay_path(SLICE) == tmp_path / "overlays" / "slice_0160_overlay.png"
    )
    assert folder.log_path(SLICE) == tmp_path / "logs" / "slice_0160.json"
