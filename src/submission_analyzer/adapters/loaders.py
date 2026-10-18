"""File adapters: everything that reads from disk lives here, not in the domain."""

from __future__ import annotations

import json
from pathlib import Path

from submission_analyzer.core.errors import DataSourceError
from submission_analyzer.core.models import Assignment

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
