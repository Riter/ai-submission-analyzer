"""Deterministic hashing of submissions and files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize to a stable string: sorted keys, no insignificant whitespace."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(
    *,
    submission_id: str,
    assignment_id: str,
    student_id: str,
    submitted_at: datetime,
    text: str,
    code: str,
    ai_used: bool,
    ai_tools: tuple[str, ...],
    ai_description: str,
) -> str:
    """SHA-256 of the normalized submission.

    The timestamp is normalized to UTC so that the same instant written with
    different offsets yields the same hash.
    """
    payload = {
        "submission_id": submission_id,
        "assignment_id": assignment_id,
        "student_id": student_id,
        "submitted_at": submitted_at.astimezone(UTC).isoformat(),
        "text": text,
        "code": code,
        "ai_declaration": {
            "used": ai_used,
            "tools": list(ai_tools),
            "description": ai_description,
        },
    }
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
