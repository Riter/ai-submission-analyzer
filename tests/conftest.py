from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from submission_analyzer.repository import InMemorySubmissionRepository
from submission_analyzer.service import IngestionService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "lab1"
ASSIGNMENT_IDS = frozenset({"lab-01-fizzbuzz", "lab-02-word-count"})

_VALID_RECORD: dict[str, Any] = {
    "submission_id": "sub-test-1",
    "assignment_id": "lab-01-fizzbuzz",
    "student_id": "student-001",
    "submitted_at": "2026-09-14T10:00:00+03:00",
    "text": "Report text.\n\nSecond paragraph.",
    "code": "def fizzbuzz(n):\n    return []\n",
    "ai_declaration": {"used": True, "tools": ["Claude"], "description": "Asked about edge cases."},
}


@pytest.fixture
def make_record() -> Callable[..., dict[str, Any]]:
    """Return a fresh valid record; keyword overrides replace top-level fields."""

    def factory(**overrides: Any) -> dict[str, Any]:
        record = copy.deepcopy(_VALID_RECORD)
        record.update(overrides)
        return record

    return factory


@pytest.fixture
def repository() -> InMemorySubmissionRepository:
    return InMemorySubmissionRepository()


@pytest.fixture
def service(repository: InMemorySubmissionRepository) -> IngestionService:
    return IngestionService(repository, ASSIGNMENT_IDS)
