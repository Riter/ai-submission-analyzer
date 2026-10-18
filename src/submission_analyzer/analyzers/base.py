"""The analyzer contract.

Any object with ``analyzer_id``, ``analyzer_version`` and a matching
``analyze`` method is an analyzer — it does not have to inherit from anything
(structural typing via :class:`typing.Protocol`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from submission_analyzer.core.config import ProcessingConfig
from submission_analyzer.core.models import Assignment, Submission
from submission_analyzer.core.reports import AnalysisReport


class SubmissionReader(Protocol):
    """Read-only access to stored submissions; analyzers never get write access."""

    def get(self, submission_id: str) -> Submission | None: ...

    def __iter__(self) -> Iterator[Submission]: ...


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    assignment: Assignment
    submissions: SubmissionReader
    config: ProcessingConfig
    config_hash: str


class Analyzer(Protocol):
    @property
    def analyzer_id(self) -> str: ...

    @property
    def analyzer_version(self) -> str: ...

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport: ...


def build_report(
    analyzer: Analyzer,
    submission: Submission,
    context: AnalysisContext,
    **fields: Any,
) -> AnalysisReport:
    """Fill in the identification fields every report must carry."""
    return AnalysisReport(
        submission_id=submission.submission_id,
        content_hash=submission.content_hash,
        analyzer_id=analyzer.analyzer_id,
        analyzer_version=analyzer.analyzer_version,
        config_hash=context.config_hash,
        **fields,
    )
