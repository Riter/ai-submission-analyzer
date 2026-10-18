"""Parsing and validation of raw submission records.

Type annotations describe what the code *expects*; they are not checked at
runtime. Records come from untrusted JSON, so every field is checked here
explicitly. The input mapping is only read, never modified.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Any

from submission_analyzer.core.errors import ErrorDetail, FormatError, SchemaError
from submission_analyzer.core.hashing import content_hash
from submission_analyzer.core.models import AIDeclaration, Submission

ID_FIELDS: tuple[str, ...] = ("submission_id", "assignment_id", "student_id")
CONTENT_FIELDS: tuple[str, ...] = ("text", "code")
REQUIRED_FIELDS: tuple[str, ...] = (*ID_FIELDS, "submitted_at", *CONTENT_FIELDS, "ai_declaration")
AI_DECLARATION_FIELDS: tuple[str, ...] = ("used", "tools", "description")


def parse_record(raw: str) -> dict[str, Any]:
    """Parse one JSONL line into a dict or raise :class:`FormatError`."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FormatError(
            [ErrorDetail("invalid_json", None, f"line is not valid JSON: {exc.msg} at column {exc.colno}")]
        ) from exc
    if not isinstance(value, dict):
        raise FormatError(
            [ErrorDetail("not_an_object", None, f"expected a JSON object, got {type(value).__name__}")]
        )
    return value


def validate_record(
    record: Mapping[str, Any],
    known_assignment_ids: Collection[str] | None = None,
) -> Submission:
    """Build a :class:`Submission` from a parsed record.

    All problems in the record are collected and raised together as one
    :class:`SchemaError`, so the user sees every issue at once.
    """
    errors: list[ErrorDetail] = []

    for key in record:
        if key not in REQUIRED_FIELDS:
            errors.append(ErrorDetail("unknown_field", key, f"field '{key}' is not part of the submission schema"))
    for key in REQUIRED_FIELDS:
        if key not in record:
            errors.append(ErrorDetail("missing_field", key, f"required field '{key}' is missing"))

    for key in (*ID_FIELDS, *CONTENT_FIELDS):
        if key in record:
            _check_string(record[key], key, errors, non_empty=key in ID_FIELDS)

    if all(isinstance(record.get(k), str) for k in CONTENT_FIELDS) and not any(
        record[k].strip() for k in CONTENT_FIELDS
    ):
        errors.append(ErrorDetail("empty_value", "text", "submission must contain non-empty text or code"))

    submitted_at = _parse_timestamp(record["submitted_at"], errors) if "submitted_at" in record else None
    declaration = _parse_ai_declaration(record["ai_declaration"], errors) if "ai_declaration" in record else None

    assignment_id = record.get("assignment_id")
    if (
        known_assignment_ids is not None
        and isinstance(assignment_id, str)
        and assignment_id.strip()
        and assignment_id not in known_assignment_ids
    ):
        errors.append(ErrorDetail("unknown_assignment", "assignment_id", f"assignment '{assignment_id}' does not exist"))

    if errors:
        raise SchemaError(errors)
    assert submitted_at is not None and declaration is not None  # guaranteed when no errors

    return Submission(
        submission_id=record["submission_id"],
        assignment_id=record["assignment_id"],
        student_id=record["student_id"],
        submitted_at=submitted_at,
        text=record["text"],
        code=record["code"],
        ai_declaration=declaration,
        content_hash=content_hash(
            submission_id=record["submission_id"],
            assignment_id=record["assignment_id"],
            student_id=record["student_id"],
            submitted_at=submitted_at,
            text=record["text"],
            code=record["code"],
            ai_used=declaration.used,
            ai_tools=declaration.tools,
            ai_description=declaration.description,
        ),
    )


def _check_string(value: Any, field: str, errors: list[ErrorDetail], *, non_empty: bool) -> bool:
    if not isinstance(value, str):
        errors.append(ErrorDetail("invalid_type", field, f"expected string, got {_json_type(value)}"))
        return False
    if non_empty and not value.strip():
        errors.append(ErrorDetail("empty_value", field, f"'{field}' must not be empty"))
        return False
    return True


def _parse_timestamp(value: Any, errors: list[ErrorDetail]) -> datetime | None:
    field = "submitted_at"
    if not isinstance(value, str):
        errors.append(ErrorDetail("invalid_type", field, f"expected ISO-8601 string, got {_json_type(value)}"))
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        errors.append(ErrorDetail("invalid_timestamp", field, f"'{value}' is not an ISO-8601 timestamp"))
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(
            ErrorDetail("naive_timestamp", field, f"'{value}' has no timezone; use e.g. '+03:00' or 'Z'")
        )
        return None
    return parsed


def _parse_ai_declaration(value: Any, errors: list[ErrorDetail]) -> AIDeclaration | None:
    prefix = "ai_declaration"
    if not isinstance(value, dict):
        errors.append(ErrorDetail("invalid_type", prefix, f"expected object, got {_json_type(value)}"))
        return None

    start = len(errors)
    for key in value:
        if key not in AI_DECLARATION_FIELDS:
            errors.append(ErrorDetail("unknown_field", f"{prefix}.{key}", f"field '{key}' is not part of ai_declaration"))
    for key in AI_DECLARATION_FIELDS:
        if key not in value:
            errors.append(ErrorDetail("missing_field", f"{prefix}.{key}", f"required field '{key}' is missing"))

    used = value.get("used")
    if "used" in value and not isinstance(used, bool):
        errors.append(ErrorDetail("invalid_type", f"{prefix}.used", f"expected boolean, got {_json_type(used)}"))

    tools = value.get("tools")
    if "tools" in value:
        if not isinstance(tools, list):
            errors.append(ErrorDetail("invalid_type", f"{prefix}.tools", f"expected array, got {_json_type(tools)}"))
        else:
            for i, tool in enumerate(tools):
                _check_string(tool, f"{prefix}.tools[{i}]", errors, non_empty=True)

    description = value.get("description")
    if "description" in value:
        _check_string(description, f"{prefix}.description", errors, non_empty=False)

    if len(errors) > start:
        return None

    # Structural consistency only. Whether a used=true declaration is
    # *sufficiently described* is an analysis concern (an analyzer reports it),
    # not a reason to reject the submission.
    if used is False and tools:
        errors.append(
            ErrorDetail("invalid_ai_declaration", f"{prefix}.tools", "tools are listed but 'used' is false")
        )
        return None

    return AIDeclaration(used=used, tools=tuple(tools), description=description)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__
