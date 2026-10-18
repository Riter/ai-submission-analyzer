"""Processing configuration: which intelligent mode (ML or LLM) is selected
and which versions of the processing pipeline are in use."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from submission_analyzer.errors import ConfigError


class Mode(StrEnum):
    ML = "ml"
    LLM = "llm"


_KNOWN_KEYS = frozenset({"mode", "processing_version", "prompt_template_version", "feature_schema_version"})


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    mode: Mode = Mode.LLM
    processing_version: str = "1.0.0"
    # LLM mode: version of the prompt template. ML mode: version of the feature schema.
    prompt_template_version: str | None = "stub-0"
    feature_schema_version: str | None = "stub-0"


def parse_config(data: Mapping[str, Any]) -> ProcessingConfig:
    """Build a config from the ``[processing]`` table of a TOML document."""
    section = data.get("processing")
    if not isinstance(section, Mapping):
        raise ConfigError("config must contain a [processing] table")

    unknown = sorted(set(section) - _KNOWN_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys in [processing]: {', '.join(unknown)}")

    raw_mode = section.get("mode", Mode.LLM.value)
    try:
        mode = Mode(raw_mode)
    except ValueError:
        allowed = ", ".join(m.value for m in Mode)
        raise ConfigError(f"processing.mode must be one of: {allowed}; got {raw_mode!r}") from None

    defaults = ProcessingConfig()
    values: dict[str, str | None] = {}
    for key in ("processing_version", "prompt_template_version", "feature_schema_version"):
        value = section.get(key, getattr(defaults, key))
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"processing.{key} must be a non-empty string")
        values[key] = value

    if mode is Mode.LLM and not values["prompt_template_version"]:
        raise ConfigError("llm mode requires processing.prompt_template_version")
    if mode is Mode.ML and not values["feature_schema_version"]:
        raise ConfigError("ml mode requires processing.feature_schema_version")

    return ProcessingConfig(mode=mode, **values)  # type: ignore[arg-type]


def load_config(path: Path) -> ProcessingConfig:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return parse_config(data)
