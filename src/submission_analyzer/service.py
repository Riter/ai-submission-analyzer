"""Submission intake service.

Pure domain logic: works with already-read lines or parsed records and never
touches files, stdin or HTTP.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from submission_analyzer.errors import ConflictError, ErrorDetail, ErrorKind, IngestionError
from submission_analyzer.models import Submission
from submission_analyzer.repository import SubmissionRepository
from submission_analyzer.validation import parse_record, validate_record


class IngestStatus(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    status: IngestStatus
    submission: Submission


@dataclass(frozen=True, slots=True)
class LineResult:
    """Outcome of one input line: either a submission or the rejection reason."""

    line_no: int
    status: IngestStatus | ErrorKind
    submission_id: str | None
    content_hash: str | None = None
    errors: tuple[ErrorDetail, ...] = ()

    @property
    def is_error(self) -> bool:
        return isinstance(self.status, ErrorKind)


@dataclass(slots=True)
class ImportReport:
    lines: list[LineResult] = field(default_factory=list)

    def _with_status(self, status: IngestStatus) -> list[LineResult]:
        return [r for r in self.lines if r.status == status]

    @property
    def accepted(self) -> list[LineResult]:
        return self._with_status(IngestStatus.ACCEPTED)

    @property
    def duplicate(self) -> list[LineResult]:
        return self._with_status(IngestStatus.DUPLICATE)

    @property
    def errors(self) -> list[LineResult]:
        return [r for r in self.lines if r.is_error]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [{"line_no": r.line_no, "submission_id": r.submission_id} for r in self.accepted],
            "duplicate": [{"line_no": r.line_no, "submission_id": r.submission_id} for r in self.duplicate],
            "errors": [
                {"line_no": r.line_no, "kind": str(r.status), "submission_id": r.submission_id, **d.to_dict()}
                for r in self.errors
                for d in r.errors
            ],
            "counts": {
                "accepted": len(self.accepted),
                "duplicate": len(self.duplicate),
                "rejected_lines": len(self.errors),
            },
        }


class IngestionService:
    def __init__(
        self,
        repository: SubmissionRepository,
        known_assignment_ids: Collection[str] | None = None,
    ) -> None:
        self._repository = repository
        self._known_assignment_ids = (
            frozenset(known_assignment_ids) if known_assignment_ids is not None else None
        )

    def submit(self, record: Mapping[str, Any]) -> IngestOutcome:
        """Validate and store one record.

        Returns ``accepted`` for a new submission and ``duplicate`` (with the
        already stored submission) for an identical resubmission.

        Raises:
            SchemaError: the record violates the schema.
            ConflictError: the id is taken by a submission with other content;
                the stored submission is left untouched.
        """
        submission = validate_record(record, self._known_assignment_ids)
        existing = self._repository.get(submission.submission_id)
        if existing is None:
            self._repository.add(submission)
            return IngestOutcome(IngestStatus.ACCEPTED, submission)
        if existing.content_hash == submission.content_hash:
            return IngestOutcome(IngestStatus.DUPLICATE, existing)
        raise ConflictError(
            [
                ErrorDetail(
                    "id_conflict",
                    "submission_id",
                    f"submission '{submission.submission_id}' already exists with different content "
                    f"({existing.content_hash[:19]}…); a new version needs a new submission_id",
                )
            ]
        )

    def import_lines(self, lines: Iterable[str], start: int = 1) -> ImportReport:
        """Import JSONL lines one by one; a bad line is reported, not fatal.

        Blank lines are skipped. Line numbers count every physical line.
        """
        report = ImportReport()
        for line_no, raw in enumerate(lines, start=start):
            if not raw.strip():
                continue
            report.lines.append(self._import_line(line_no, raw))
        return report

    def _import_line(self, line_no: int, raw: str) -> LineResult:
        submission_id: str | None = None
        try:
            record = parse_record(raw)
            candidate = record.get("submission_id")
            submission_id = candidate if isinstance(candidate, str) else None
            outcome = self.submit(record)
        except IngestionError as exc:
            return LineResult(line_no, exc.kind, submission_id, errors=exc.details)
        return LineResult(
            line_no,
            outcome.status,
            outcome.submission.submission_id,
            content_hash=outcome.submission.content_hash,
        )
