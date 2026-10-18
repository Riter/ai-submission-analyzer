from __future__ import annotations

import json

import pytest

from submission_analyzer.errors import ConflictError, ErrorKind, SchemaError
from submission_analyzer.service import IngestStatus


def test_new_submission_is_accepted(service, repository, make_record):
    outcome = service.submit(make_record())

    assert outcome.status is IngestStatus.ACCEPTED
    assert repository.get("sub-test-1") == outcome.submission
    assert len(repository) == 1


def test_duplicate_returns_stored_submission_and_does_not_grow(service, repository, make_record):
    first = service.submit(make_record())
    second = service.submit(make_record())

    assert second.status is IngestStatus.DUPLICATE
    assert second.submission is first.submission
    assert len(repository) == 1


def test_duplicate_detected_across_timezone_notation(service, repository, make_record):
    service.submit(make_record(submitted_at="2026-09-14T10:00:00+03:00"))
    outcome = service.submit(make_record(submitted_at="2026-09-14T07:00:00Z"))
    assert outcome.status is IngestStatus.DUPLICATE
    assert len(repository) == 1


def test_conflict_does_not_overwrite_original(service, repository, make_record):
    original = service.submit(make_record()).submission

    with pytest.raises(ConflictError) as exc_info:
        service.submit(make_record(code="print('other content')\n"))

    detail = exc_info.value.details[0]
    assert exc_info.value.kind is ErrorKind.CONFLICT
    assert (detail.code, detail.field) == ("id_conflict", "submission_id")
    assert repository.get("sub-test-1") is original
    assert len(repository) == 1


def test_new_version_with_new_id_is_accepted(service, repository, make_record):
    service.submit(make_record())
    outcome = service.submit(make_record(submission_id="sub-test-2", code="print('v2')\n"))

    assert outcome.status is IngestStatus.ACCEPTED
    assert len(repository) == 2


def test_invalid_record_is_not_stored(service, repository, make_record):
    with pytest.raises(SchemaError):
        service.submit(make_record(student_id=""))
    assert len(repository) == 0


def test_import_continues_after_corrupted_line(service, repository, make_record):
    lines = [
        json.dumps(make_record(submission_id="a")),
        '{"submission_id": "broken", ',
        "[1, 2]",
        json.dumps(make_record(submission_id="b")),
    ]
    report = service.import_lines(lines)

    assert [r.submission_id for r in report.accepted] == ["a", "b"]
    assert [(r.line_no, r.status) for r in report.errors] == [(2, ErrorKind.FORMAT), (3, ErrorKind.FORMAT)]
    assert len(repository) == 2


def test_import_reports_line_numbers_and_skips_blank_lines(service, make_record):
    lines = [
        json.dumps(make_record(submission_id="a")),
        "",
        "   \n",
        json.dumps(make_record(submission_id="b", student_id="")),
    ]
    report = service.import_lines(lines)

    assert len(report.lines) == 2
    error = report.errors[0]
    assert error.line_no == 4
    assert error.submission_id == "b"
    assert error.status is ErrorKind.SCHEMA


def test_import_distinguishes_duplicate_and_conflict(service, make_record):
    lines = [
        json.dumps(make_record()),
        json.dumps(make_record()),
        json.dumps(make_record(code="changed\n")),
    ]
    report = service.import_lines(lines)
    data = report.to_dict()

    assert data["counts"] == {"accepted": 1, "duplicate": 1, "rejected_lines": 1}
    assert data["duplicate"] == [{"line_no": 2, "submission_id": "sub-test-1"}]
    assert data["errors"][0]["kind"] == "conflict"
    assert data["errors"][0]["code"] == "id_conflict"


def test_report_flattens_every_error_detail(service, make_record):
    report = service.import_lines([json.dumps(make_record(student_id="", extra=1))])
    errors = report.to_dict()["errors"]

    assert {(e["code"], e["field"]) for e in errors} == {("empty_value", "student_id"), ("unknown_field", "extra")}
    assert all(e["line_no"] == 1 and e["kind"] == "schema" for e in errors)
