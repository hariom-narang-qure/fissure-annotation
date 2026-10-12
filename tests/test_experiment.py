import threading

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from fissure_annotation.experiment import SliceExport, WaypointExperiment, export_slices, run_experiment, write_sheets
from fissure_annotation.imaging.png import save_png
from fissure_annotation.settings import load_settings
from test_agent import FINISH, TRACE, FakeModel


@pytest.fixture
def config(tmp_path):
    for z in (1, 2, 3):
        image = np.full((512, 512), 50, dtype=np.uint8)
        image[300, 100:400] = 220
        save_png(image, tmp_path / "slices" / f"slice_000{z}.png")
    return WaypointExperiment(slices_dir=tmp_path / "slices", out_dir=tmp_path / "run", first_slice=1, last_slice=3)


def test_the_slice_range_is_inclusive(config):
    assert config.slice_indices == [1, 2, 3]


def test_the_yaml_config_loads_with_the_agent_defaults(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("slices_dir: s\nout_dir: o\nfirst_slice: 153\nlast_slice: 167\n")
    config = load_settings(WaypointExperiment, ["--config", str(path), "--agent.max-steps", "20"])
    assert (len(config.slice_indices), config.agent.max_steps, config.agent.model) == (15, 20, "@openai/gpt-6.1-sol")


class ModelPerSlice:
    "every slice runs in its own thread and starts without a previous_response_id, so each gets a fresh scripted model"

    def __init__(self):
        self.local, self.responses = threading.local(), self

    def create(self, **request):
        if "previous_response_id" not in request:
            self.local.model = FakeModel([TRACE], [FINISH])
        return self.local.model.create(**request)


def test_every_slice_gets_a_log_and_the_contact_sheet_has_one_tile_per_slice(config):
    logs = run_experiment(config, ModelPerSlice())
    assert [log.traces for log in logs] == [1, 1, 1]
    x0, y0, x1, y1 = config.agent.overlay_box
    assert Image.open(config.out_dir / "contact_sheet.png").size == (3 * (x1 - x0), y1 - y0)


def test_the_side_by_side_sheet_pairs_each_slice_with_its_annotation(config):
    run_experiment(config, ModelPerSlice())
    x0, y0, x1, y1 = config.agent.overlay_box
    width, height = x1 - x0, y1 - y0
    assert Image.open(config.out_dir / "side_by_side.png").size == (2 * 2 * width, 2 * height)  # 3 pairs, 2 per row


def test_the_sheets_can_be_remade_from_a_finished_run_without_a_model(config):
    run_experiment(config, ModelPerSlice())
    (config.out_dir / "side_by_side.png").unlink()
    write_sheets(config)
    assert (config.out_dir / "side_by_side.png").exists()


def test_export_slices_writes_pngs(tmp_path):
    nifti = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.full((12, 10, 2), -850, dtype=np.int16), np.eye(4)), nifti)
    assert export_slices(SliceExport(nifti=nifti, out_dir=tmp_path / "out")) == 2
