"""Explicit registry of analyzer factories.

The analysis service looks analyzers up by id here. Adding an analyzer means
registering one more factory — the service itself does not change.
"""

from __future__ import annotations

from collections.abc import Callable

from submission_analyzer.analyzers import (
    Analyzer,
    LengthAnalyzer,
    ModelStubAnalyzer,
    RequirementsAnalyzer,
    TemplateAnalyzer,
)
from submission_analyzer.core.config import ProcessingConfig
from submission_analyzer.core.errors import ConfigError

AnalyzerFactory = Callable[[ProcessingConfig], Analyzer]


class AnalyzerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, AnalyzerFactory] = {}

    def register(self, analyzer_id: str, factory: AnalyzerFactory) -> None:
        if analyzer_id in self._factories:
            raise ValueError(f"analyzer '{analyzer_id}' is already registered")
        self._factories[analyzer_id] = factory

    def ids(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def create(self, analyzer_id: str, config: ProcessingConfig) -> Analyzer:
        factory = self._factories.get(analyzer_id)
        if factory is None:
            known = ", ".join(sorted(self._factories)) or "none"
            raise ConfigError(f"unknown analyzer '{analyzer_id}'; registered: {known}")
        analyzer = factory(config)
        if analyzer.analyzer_id != analyzer_id:
            raise ConfigError(
                f"factory registered as '{analyzer_id}' produced analyzer '{analyzer.analyzer_id}'"
            )
        return analyzer


def default_registry() -> AnalyzerRegistry:
    registry = AnalyzerRegistry()
    registry.register("length", lambda config: LengthAnalyzer())
    registry.register("template", lambda config: TemplateAnalyzer())
    registry.register("requirements", lambda config: RequirementsAnalyzer())
    registry.register("model_stub", ModelStubAnalyzer)
    return registry
