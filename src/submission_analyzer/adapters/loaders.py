"""File adapters: everything that reads from disk lives here, not in the domain."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from submission_analyzer.core.corpus import (
    DerivationRelation,
    LabeledPair,
    Origin,
    Provenance,
    Split,
    SplitName,
)
from submission_analyzer.core.errors import DataSourceError
from submission_analyzer.core.models import Assignment

T = TypeVar("T")

_ASSIGNMENT_FIELDS = frozenset({"assignment_id", "title", "requirements", "allowed_template"})


def load_assignments(path: Path) -> list[Assignment]:
    """Load ``assignments.json``: ``{"assignments": [ {...}, ... ]}``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataSourceError(f"cannot read {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"{path} is not valid JSON: {exc.msg} (line {exc.lineno})") from exc

    items = data.get("assignments") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise DataSourceError(f"{path}: expected an object with an 'assignments' array")

    result: list[Assignment] = []
    for i, item in enumerate(items):
        where = f"{path}: assignments[{i}]"
        if not isinstance(item, dict) or set(item) != _ASSIGNMENT_FIELDS:
            raise DataSourceError(f"{where} must have exactly the fields {sorted(_ASSIGNMENT_FIELDS)}")
        requirements = item["requirements"]
        if not (
            isinstance(item["assignment_id"], str)
            and item["assignment_id"].strip()
            and isinstance(item["title"], str)
            and isinstance(item["allowed_template"], str)
            and isinstance(requirements, list)
            and all(isinstance(r, str) for r in requirements)
        ):
            raise DataSourceError(f"{where} has fields of wrong type")
        result.append(
            Assignment(
                assignment_id=item["assignment_id"],
                title=item["title"],
                requirements=tuple(requirements),
                allowed_template=item["allowed_template"],
            )
        )

    ids = [a.assignment_id for a in result]
    if len(ids) != len(set(ids)):
        raise DataSourceError(f"{path}: duplicate assignment_id")
    return result


_PROVENANCE_FIELDS = frozenset({"submission_id", "family_id", "origin", "derived_from"})
_PAIR_FIELDS = frozenset(
    {"a", "b", "assignment_id", "family_a", "family_b", "derivation_relation", "template_only", "split"}
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataSourceError(f"cannot read {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"{path} is not valid JSON: {exc.msg} (line {exc.lineno})") from exc


def _read_jsonl(path: Path, fields: frozenset[str], build: Callable[[dict[str, Any]], T]) -> list[T]:
    """Read a metadata JSONL file; unlike submissions, any bad line is fatal (it is our own data)."""
    try:
        fh = path.open(encoding="utf-8")
    except OSError as exc:
        raise DataSourceError(f"cannot read {path}: {exc.strerror}") from exc
    result: list[T] = []
    with fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_no}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataSourceError(f"{where}: invalid JSON: {exc.msg}") from exc
            if not isinstance(item, dict) or set(item) != fields:
                raise DataSourceError(f"{where}: expected exactly the fields {sorted(fields)}")
            try:
                result.append(build(item))
            except (TypeError, ValueError) as exc:
                raise DataSourceError(f"{where}: {exc}") from exc
    return result


def _str(item: dict[str, Any], key: str) -> str:
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value


def load_provenance(path: Path) -> list[Provenance]:
    def build(item: dict[str, Any]) -> Provenance:
        derived_from = item["derived_from"]
        if derived_from is not None and not isinstance(derived_from, str):
            raise ValueError("'derived_from' must be a string or null")
        return Provenance(_str(item, "submission_id"), _str(item, "family_id"), Origin(item["origin"]), derived_from)

    return _read_jsonl(path, _PROVENANCE_FIELDS, build)


def load_pairs(path: Path) -> list[LabeledPair]:
    def build(item: dict[str, Any]) -> LabeledPair:
        if not isinstance(item["template_only"], bool):
            raise ValueError("'template_only' must be true or false")
        return LabeledPair(
            a=_str(item, "a"),
            b=_str(item, "b"),
            assignment_id=_str(item, "assignment_id"),
            family_a=_str(item, "family_a"),
            family_b=_str(item, "family_b"),
            derivation_relation=DerivationRelation(item["derivation_relation"]),
            template_only=item["template_only"],
            split=SplitName(item["split"]),
        )

    return _read_jsonl(path, _PAIR_FIELDS, build)


def load_split(path: Path) -> Split:
    """Load ``split.json``: ``{"families": {family: side}, "submissions": {id: side}}``."""
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("families"), dict) or not isinstance(
        data.get("submissions"), dict
    ):
        raise DataSourceError(f"{path}: expected an object with 'families' and 'submissions' maps")
    try:
        families = {str(k): SplitName(v) for k, v in data["families"].items()}
        submissions = {str(k): SplitName(v) for k, v in data["submissions"].items()}
    except ValueError as exc:
        raise DataSourceError(f"{path}: {exc}") from exc
    return Split(families, submissions)
