"""Load a config model from a YAML file with command line flags on top.

- The YAML (`--config`) holds what distinguishes one experiment from another, so it lives in git.
- Flags are for one-off changes on top of it, and nested sections use dotted flags such as `--agent.use-tracer false`.
- Environment variables are switched off, otherwise a variable named `MODEL` would silently override a field.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ExperimentSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_kebab_case=True, frozen=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        "the config file arrives as init values, and nothing else is read"
        return (init_settings,)


def config_path_from(argv: Sequence[str]) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, _ = parser.parse_known_args(argv)
    return known.config


def without_config_flag(argv: Sequence[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--config":
            skip = True
        elif not a.startswith("--config="):
            out.append(a)
    return out


def load_settings[T: ExperimentSettings](cls: type[T], argv: Sequence[str]) -> T:
    """The file is read first and handed over as init values, with the flags layered on top.

    The settings class cannot read the path from the flags it is parsing, so the path is found first.
    """
    path = config_path_from(argv)
    values = yaml.safe_load(path.read_text()) if path else {}
    return cls(_cli_parse_args=without_config_flag(argv), **(values or {}))
