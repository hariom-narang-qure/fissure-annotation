import pytest
from pydantic import ValidationError

from fissure_annotation.scroll_experiment import ScrollExperimentConfig
from fissure_annotation.settings import load_settings


@pytest.fixture
def yaml_file(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("cases: my_cases.json\nonly: scan2\nagent:\n  use_tracer: false\n  max_steps: 125\n")
    return path


def test_values_come_from_the_yaml(yaml_file):
    config = load_settings(ScrollExperimentConfig, ["--config", str(yaml_file)])
    assert (config.only, config.agent.use_tracer, config.agent.max_steps) == ("scan2", False, 125)


def test_a_flag_wins_over_the_yaml(yaml_file):
    config = load_settings(ScrollExperimentConfig, ["--config", str(yaml_file), "--only", "scan9"])
    assert config.only == "scan9"


def test_overriding_one_key_of_a_section_keeps_the_files_other_keys(yaml_file):
    config = load_settings(ScrollExperimentConfig, ["--config", str(yaml_file), "--agent.plain-images", "true"])
    assert (config.agent.plain_images, config.agent.use_tracer, config.agent.max_steps) == (True, False, 125)


def test_the_config_flag_can_be_written_with_an_equals_sign(yaml_file):
    assert load_settings(ScrollExperimentConfig, [f"--config={yaml_file}"]).only == "scan2"


def test_no_file_means_the_defaults():
    assert load_settings(ScrollExperimentConfig, []).agent.use_tracer is True


def test_a_typo_in_the_yaml_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("agent:\n  use_tracr: false\n")
    with pytest.raises(ValidationError):
        load_settings(ScrollExperimentConfig, ["--config", str(path)])


def test_environment_variables_do_not_override_fields(yaml_file, monkeypatch):
    monkeypatch.setenv("ONLY", "from_env")
    assert load_settings(ScrollExperimentConfig, ["--config", str(yaml_file)]).only == "scan2"
