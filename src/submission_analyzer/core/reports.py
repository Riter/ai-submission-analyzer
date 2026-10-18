"""Analysis report — the result contract shared by every analyzer.

A report describes *signals* (similarity, criterion statuses, evidence). It is
never a verdict: decisions are made by a teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from submission_analyzer.core.models import Submission
from submission_analyzer.core.textutils import source_lines


class ReportStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CriterionStatus(StrEnum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceSource(StrEnum):
    TEXT = "text"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class Evidence:
    """A range of lines in the original submission content (1-based, inclusive)."""

    source: EvidenceSource
    start_line: int
    end_line: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RubricFeedback:
    """Feedback on one criterion. ``status`` is ``None`` for purely informational notes."""

    criterion_id: str
    status: CriterionStatus | None
    explanation: str
    evidence: tuple[Evidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "status": str(self.status) if self.status is not None else None,
            "explanation": self.explanation,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    submission_id: str
    content_hash: str
    analyzer_id: str
    analyzer_version: str
    config_hash: str
    status: ReportStatus = ReportStatus.SUCCEEDED
    score: float | None = None
    similarity: float | None = None
    rubric_feedback: tuple[RubricFeedback, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    processing_error: str | None = None

    def all_evidence(self) -> tuple[Evidence, ...]:
        return self.evidence + tuple(e for fb in self.rubric_feedback for e in fb.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "content_hash": self.content_hash,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "config_hash": self.config_hash,
            "status": str(self.status),
            "score": self.score,
            "similarity": self.similarity,
            "rubric_feedback": [fb.to_dict() for fb in self.rubric_feedback],
            "evidence": [e.to_dict() for e in self.evidence],
            "processing_error": self.processing_error,
        }


def invalid_evidence(report: AnalysisReport, submission: Submission) -> list[str]:
    """Return a description of every evidence range that does not point to existing lines."""
    line_counts = {
        EvidenceSource.TEXT: len(source_lines(submission.text)),
        EvidenceSource.CODE: len(source_lines(submission.code)),
    }
    problems: list[str] = []
    for ev in report.all_evidence():
        limit = line_counts[ev.source]
        if not 1 <= ev.start_line <= ev.end_line <= limit:
            problems.append(f"{ev.source} lines {ev.start_line}-{ev.end_line} outside 1-{limit}")
    return problems
