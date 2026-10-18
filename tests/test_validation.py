from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest

from submission_analyzer.core.errors import ErrorKind, FormatError, SchemaError
from submission_analyzer.ingestion.validation import parse_record, validate_record

from .conftest import ASSIGNMENT_IDS


def codes(exc: SchemaError) -> set[tuple[str, str | None]]:
    return {(d.code, d.field) for d in exc.details}


def test_valid_record_is_accepted(make_record):
    submission = validate_record(make_record(), ASSIGNMENT_IDS)

    assert submission.submission_id == "sub-test-1"
    assert submission.submitted_at == datetime(2026, 9, 14, 7, 0, tzinfo=UTC)
    assert submission.ai_declaration.tools == ("Claude",)
    assert submission.content_hash.startswith("sha256:")


@pytest.mark.parametrize("raw", ['{"submission_id": "x", ', "not json", "{'single': 'quotes'}"])
def test_corrupted_json_is_format_error(raw):
    with pytest.raises(FormatError) as exc_info:
        parse_record(raw)
    assert exc_info.value.kind is ErrorKind.FORMAT
    assert exc_info.value.details[0].code == "invalid_json"


@pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "42", "null"])
def test_non_object_json_is_format_error(raw):
    with pytest.raises(FormatError) as exc_info:
        parse_record(raw)
    assert exc_info.value.details[0].code == "not_an_object"


@pytest.mark.parametrize(
    "timestamp",
    ["2026-13-45T25:00:00+03:00", "yesterday", "", "14.09.2026 10:00"],
)
def test_invalid_timestamp(make_record, timestamp):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(submitted_at=timestamp))
    assert codes(exc_info.value) == {("invalid_timestamp", "submitted_at")}


@pytest.mark.parametrize("timestamp", ["2026-09-14T10:00:00", "2026-09-14"])
def test_timestamp_without_timezone_is_rejected(make_record, timestamp):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(submitted_at=timestamp))
    assert codes(exc_info.value) == {("naive_timestamp", "submitted_at")}


def test_utc_z_suffix_is_accepted(make_record):
    submission = validate_record(make_record(submitted_at="2026-09-14T07:00:00Z"))
    assert submission.submitted_at.utcoffset() is not None


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_empty_student_id(make_record, value):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(student_id=value))
    detail = exc_info.value.details[0]
    assert (detail.code, detail.field) == ("empty_value", "student_id")
    assert "student_id" in detail.message


def test_unknown_field(make_record):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(grade=5))
    assert codes(exc_info.value) == {("unknown_field", "grade")}


def test_missing_fields_are_all_reported(make_record):
    record = make_record()
    del record["code"]
    del record["ai_declaration"]
    with pytest.raises(SchemaError) as exc_info:
        validate_record(record)
    assert codes(exc_info.value) == {("missing_field", "code"), ("missing_field", "ai_declaration")}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submission_id", 1),
        ("student_id", None),
        ("assignment_id", ["lab"]),
        ("text", 3.14),
        ("code", {"main": "x"}),
        ("submitted_at", 1726300000),
        ("ai_declaration", "used"),
    ],
)
def test_wrong_top_level_types(make_record, field, value):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(**{field: value}))
    assert ("invalid_type", field) in codes(exc_info.value)


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"used": "yes", "tools": [], "description": ""}, ("invalid_type", "ai_declaration.used")),
        ({"used": 1, "tools": [], "description": ""}, ("invalid_type", "ai_declaration.used")),
        ({"used": True, "tools": "Claude", "description": ""}, ("invalid_type", "ai_declaration.tools")),
        ({"used": True, "tools": [""], "description": ""}, ("empty_value", "ai_declaration.tools[0]")),
        ({"used": True, "tools": [], "description": None}, ("invalid_type", "ai_declaration.description")),
        ({"used": True, "tools": []}, ("missing_field", "ai_declaration.description")),
        ({"used": True, "tools": [], "description": "", "model": "x"}, ("unknown_field", "ai_declaration.model")),
        ({"used": False, "tools": ["ChatGPT"], "description": ""}, ("invalid_ai_declaration", "ai_declaration.tools")),
    ],
)
def test_ai_declaration_validation(make_record, declaration, expected):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(ai_declaration=declaration))
    assert expected in codes(exc_info.value)


def test_used_true_with_empty_description_is_structurally_valid(make_record):
    # Insufficient description is reported by an analyzer, not rejected at intake.
    record = make_record(ai_declaration={"used": True, "tools": [], "description": ""})
    assert validate_record(record).ai_declaration.used is True


def test_submission_needs_some_content(make_record):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(text=" ", code="\n"))
    assert codes(exc_info.value) == {("empty_value", "text")}


def test_unknown_assignment(make_record):
    with pytest.raises(SchemaError) as exc_info:
        validate_record(make_record(assignment_id="lab-99"), ASSIGNMENT_IDS)
    assert codes(exc_info.value) == {("unknown_assignment", "assignment_id")}


def test_all_errors_of_a_record_are_collected(make_record):
    record = make_record(student_id="", submitted_at="2026-09-14T10:00:00", extra=1)
    with pytest.raises(SchemaError) as exc_info:
        validate_record(record)
    assert codes(exc_info.value) == {
        ("empty_value", "student_id"),
        ("naive_timestamp", "submitted_at"),
        ("unknown_field", "extra"),
    }


@pytest.mark.parametrize(
    "overrides",
    [{}, {"student_id": ""}, {"grade": 1}, {"ai_declaration": {"used": "x", "tools": [1]}}],
)
def test_input_record_is_not_mutated(make_record, overrides):
    record = make_record(**overrides)
    snapshot = copy.deepcopy(record)
    try:
        validate_record(record, ASSIGNMENT_IDS)
    except SchemaError:
        pass
    assert record == snapshot


def test_submission_is_immutable(make_record):
    submission = validate_record(make_record())
    with pytest.raises(AttributeError):
        submission.code = "print('changed')"  # type: ignore[misc]


def test_same_instant_in_other_timezone_has_same_hash(make_record):
    a = validate_record(make_record(submitted_at="2026-09-14T10:00:00+03:00"))
    b = validate_record(make_record(submitted_at="2026-09-14T07:00:00+00:00"))
    assert a.content_hash == b.content_hash


def test_content_change_changes_hash(make_record):
    a = validate_record(make_record())
    b = validate_record(make_record(code="def fizzbuzz(n):\n    return ['1']\n"))
    assert a.content_hash != b.content_hash
