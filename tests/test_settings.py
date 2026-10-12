import pytest
from pydantic import ValidationError

from fissure_annotation.settings import ExperimentSettings, load_settings


class Agent(ExperimentSettings):
    max_steps: int = 14
    reasoning: str = "medium"


class Config(ExperimentSettings):
    scan: str = "default.nii.gz"
    agent: Agent = Agent()


@pytest.fixture
def yaml_file(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("scan: a.nii.gz\nagent:\n  max_steps: 20\n")
    return path


def test_values_come_from_the_yaml(yaml_file):
    config = load_settings(Config, ["--config", str(yaml_file)])
    assert (config.scan, config.agent.max_steps) == ("a.nii.gz", 20)


def test_a_flag_wins_over_the_yaml(yaml_file):
    assert load_settings(Config, ["--config", str(yaml_file), "--scan", "b.nii.gz"]).scan == "b.nii.gz"


def test_overriding_one_key_of_a_section_keeps_the_files_other_keys(yaml_file):
    config = load_settings(Config, ["--config", str(yaml_file), "--agent.reasoning", "high"])
    assert (config.agent.reasoning, config.agent.max_steps) == ("high", 20)


def test_the_config_flag_can_be_written_with_an_equals_sign(yaml_file):
    assert load_settings(Config, [f"--config={yaml_file}"]).scan == "a.nii.gz"


def test_no_file_means_the_defaults():
    assert load_settings(Config, []).agent.max_steps == 14


def test_a_typo_in_the_yaml_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("agent:\n  max_stepz: 3\n")
    with pytest.raises(ValidationError):
        load_settings(Config, ["--config", str(path)])


def test_environment_variables_do_not_override_fields(yaml_file, monkeypatch):
    monkeypatch.setenv("SCAN", "from_env")
    assert load_settings(Config, ["--config", str(yaml_file)]).scan == "a.nii.gz"
