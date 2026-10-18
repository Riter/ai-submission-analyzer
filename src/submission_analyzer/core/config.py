"""Processing configuration: which intelligent mode (ML or LLM) is selected,
which analyzers run, and which versions of the pipeline are in use."""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from submission_analyzer.core.errors import ConfigError
from submission_analyzer.core.hashing import canonical_json


class Mode(StrEnum):
    ML = "ml"
    LLM = "llm"


_KNOWN_TABLES = frozenset({"processing", "analyzers"})
_KNOWN_KEYS = frozenset({"mode", "processing_version", "prompt_template_version", "feature_schema_version"})
DEFAULT_ANALYZERS: tuple[str, ...] = ("length", "template", "requirements", "model_stub")


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    mode: Mode = Mode.LLM
    processing_version: str = "1.0.0"
    # LLM mode: version of the prompt template. ML mode: version of the feature schema.
    prompt_template_version: str | None = "stub-0"
    feature_schema_version: str | None = "stub-0"
    analyzers: tuple[str, ...] = DEFAULT_ANALYZERS


def config_hash(config: ProcessingConfig) -> str:
    """Stable fingerprint of everything that can change analysis results.

    Two runs with equal ``config_hash`` used the same mode, versions and
    analyzer set, so their reports are comparable.
    """
    payload = {
        "mode": str(config.mode),
        "processing_version": config.processing_version,
        "prompt_template_version": config.prompt_template_version,
        "feature_schema_version": config.feature_schema_version,
        "analyzers": list(config.analyzers),
    }
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def parse_config(data: Mapping[str, Any]) -> ProcessingConfig:
    """Build a config from a parsed TOML document (``[processing]`` and optional ``[analyzers]``)."""
    unknown_tables = sorted(set(data) - _KNOWN_TABLES)
    if unknown_tables:
        raise ConfigError(f"unknown config tables: {', '.join(unknown_tables)}")

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

    return ProcessingConfig(mode=mode, analyzers=_parse_analyzers(data.get("analyzers")), **values)  # type: ignore[arg-type]


def _parse_analyzers(section: Any) -> tuple[str, ...]:
    if section is None:
        return DEFAULT_ANALYZERS
    if not isinstance(section, Mapping) or set(section) != {"enabled"}:
        raise ConfigError("[analyzers] must contain exactly one key: enabled")
    enabled = section["enabled"]
    if (
        not isinstance(enabled, list)
        or not enabled
        or not all(isinstance(a, str) and a.strip() for a in enabled)
    ):
        raise ConfigError("analyzers.enabled must be a non-empty list of analyzer ids")
    if len(set(enabled)) != len(enabled):
        raise ConfigError("analyzers.enabled contains duplicate ids")
    return tuple(enabled)


def load_config(path: Path) -> ProcessingConfig:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return parse_config(data)
