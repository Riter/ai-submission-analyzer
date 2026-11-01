"""Processing configuration: which intelligent mode (ML or LLM) is selected,
which analyzers run, how similarity is computed, and which versions of the
pipeline are in use."""

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


_KNOWN_TABLES = frozenset({"processing", "analyzers", "similarity", "experiment"})
_KNOWN_KEYS = frozenset({"mode", "processing_version", "prompt_template_version", "feature_schema_version"})
_SIMILARITY_KEYS = frozenset({"ngram", "normalize_identifiers", "top_k", "methods"})
_EXPERIMENT_KEYS = frozenset({"seed", "classifier"})
CLASSIFIERS: tuple[str, ...] = ("logistic_regression",)
DEFAULT_ANALYZERS: tuple[str, ...] = ("length", "template", "requirements", "token_similarity", "model_stub")
SIMILARITY_METHODS: tuple[str, ...] = ("jaccard", "tfidf")


@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    """How code is turned into features and compared (lab 2)."""

    ngram: int = 3
    # Replace user-defined names with ``ID`` and string literals with ``STR``.
    normalize_identifiers: bool = True
    top_k: int = 5
    methods: tuple[str, ...] = SIMILARITY_METHODS

    def to_dict(self) -> dict[str, Any]:
        return {
            "ngram": self.ngram,
            "normalize_identifiers": self.normalize_identifiers,
            "top_k": self.top_k,
            "methods": list(self.methods),
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Evaluation experiment of the ML track (lab 2, stage B)."""

    seed: int = 42
    classifier: str = "logistic_regression"

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "classifier": self.classifier}


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    mode: Mode = Mode.ML
    processing_version: str = "1.0.0"
    # LLM mode: version of the prompt template. ML mode: version of the feature schema.
    prompt_template_version: str | None = "stub-0"
    feature_schema_version: str | None = "pair-1"
    analyzers: tuple[str, ...] = DEFAULT_ANALYZERS
    similarity: SimilarityConfig = SimilarityConfig()
    experiment: ExperimentConfig = ExperimentConfig()


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
        "similarity": config.similarity.to_dict(),
        "experiment": config.experiment.to_dict(),
    }
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def parse_config(data: Mapping[str, Any]) -> ProcessingConfig:
    """Build a config from a parsed TOML document.

    ``[processing]`` is required; ``[analyzers]``, ``[similarity]`` and ``[experiment]`` are optional.
    """
    unknown_tables = sorted(set(data) - _KNOWN_TABLES)
    if unknown_tables:
        raise ConfigError(f"unknown config tables: {', '.join(unknown_tables)}")

    section = data.get("processing")
    if not isinstance(section, Mapping):
        raise ConfigError("config must contain a [processing] table")

    unknown = sorted(set(section) - _KNOWN_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys in [processing]: {', '.join(unknown)}")

    defaults = ProcessingConfig()
    raw_mode = section.get("mode", defaults.mode.value)
    try:
        mode = Mode(raw_mode)
    except ValueError:
        allowed = ", ".join(m.value for m in Mode)
        raise ConfigError(f"processing.mode must be one of: {allowed}; got {raw_mode!r}") from None

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

    return ProcessingConfig(  # type: ignore[arg-type]
        mode=mode,
        analyzers=_parse_analyzers(data.get("analyzers")),
        similarity=_parse_similarity(data.get("similarity")),
        experiment=_parse_experiment(data.get("experiment")),
        **values,
    )


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


def _parse_similarity(section: Any) -> SimilarityConfig:
    defaults = SimilarityConfig()
    if section is None:
        return defaults
    if not isinstance(section, Mapping):
        raise ConfigError("[similarity] must be a table")
    unknown = sorted(set(section) - _SIMILARITY_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys in [similarity]: {', '.join(unknown)}")

    def positive_int(key: str) -> int:
        value = section.get(key, getattr(defaults, key))
        # bool is a subclass of int; `ngram = true` must not pass as 1.
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"similarity.{key} must be a positive integer")
        return value

    normalize = section.get("normalize_identifiers", defaults.normalize_identifiers)
    if not isinstance(normalize, bool):
        raise ConfigError("similarity.normalize_identifiers must be true or false")

    methods = section.get("methods", list(defaults.methods))
    if not isinstance(methods, list) or not methods or not all(isinstance(m, str) for m in methods):
        raise ConfigError("similarity.methods must be a non-empty list of method ids")
    unknown_methods = sorted(set(methods) - set(SIMILARITY_METHODS))
    if unknown_methods:
        raise ConfigError(
            f"unknown similarity methods: {', '.join(unknown_methods)}; known: {', '.join(SIMILARITY_METHODS)}"
        )
    if len(set(methods)) != len(methods):
        raise ConfigError("similarity.methods contains duplicate ids")

    return SimilarityConfig(
        ngram=positive_int("ngram"),
        normalize_identifiers=normalize,
        top_k=positive_int("top_k"),
        methods=tuple(methods),
    )


def _parse_experiment(section: Any) -> ExperimentConfig:
    defaults = ExperimentConfig()
    if section is None:
        return defaults
    if not isinstance(section, Mapping):
        raise ConfigError("[experiment] must be a table")
    unknown = sorted(set(section) - _EXPERIMENT_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys in [experiment]: {', '.join(unknown)}")
    seed = section.get("seed", defaults.seed)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("experiment.seed must be a non-negative integer")
    classifier = section.get("classifier", defaults.classifier)
    if classifier not in CLASSIFIERS:
        raise ConfigError(f"experiment.classifier must be one of: {', '.join(CLASSIFIERS)}; got {classifier!r}")
    return ExperimentConfig(seed=seed, classifier=classifier)


def load_config(path: Path) -> ProcessingConfig:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return parse_config(data)
