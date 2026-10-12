"""Every YAML in configs/ must load into the settings class its script uses, so a renamed field fails here and not on a long run."""
from pathlib import Path

import pytest

from fissure_annotation.experiment import OneShotConfig
from fissure_annotation.scan_experiment import ScanExperimentConfig
from fissure_annotation.scroll_experiment import ScrollExperimentConfig
from fissure_annotation.settings import load_settings
from fissure_annotation.slice_experiment import SliceExperimentConfig

CONFIGS = Path(__file__).parents[1] / "configs"
SETTINGS_OF = {"scroll": ScrollExperimentConfig, "scan": ScanExperimentConfig, "one_shot": OneShotConfig, "slice": SliceExperimentConfig}


def settings_class_for(path: Path):
    return next(cls for prefix, cls in SETTINGS_OF.items() if path.name.startswith(prefix))


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_config_loads(path):
    load_settings(settings_class_for(path), ["--config", str(path)])


def test_the_smoke_config_runs_only_a_couple_of_jobs():
    config = load_settings(ScrollExperimentConfig, ["--config", str(CONFIGS / "scroll_smoke.yaml")])
    assert config.max_jobs == 2 and config.workers == 2


def test_one_shot_defaults_to_more_workers_than_the_agents():
    assert OneShotConfig().workers == 8 and ScrollExperimentConfig().workers == 5
