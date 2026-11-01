from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from submission_analyzer.ingestion.service import IngestionService
from submission_analyzer.storage.repository import InMemorySubmissionRepository

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "lab1"
LAB2_DIR = PROJECT_ROOT / "datasets" / "lab2"
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


TEMPLATE = "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n    return result\n"


@pytest.fixture
def assignment():
    from submission_analyzer.core.models import Assignment

    return Assignment(
        assignment_id="lab-01-fizzbuzz",
        title="FizzBuzz",
        requirements=(
            "Function fizzbuzz(n) returns a list of strings for 1..n",
            "Multiples of 3 are 'Fizz', of 5 are 'Buzz', of both are 'FizzBuzz'",
            "Report explains the approach",
        ),
        allowed_template=TEMPLATE,
    )


@pytest.fixture
def make_context(assignment, repository):
    """Build an AnalysisContext for a given config (default: ProcessingConfig())."""
    from submission_analyzer.analyzers import AnalysisContext
    from submission_analyzer.core.config import ProcessingConfig, config_hash

    def factory(config: ProcessingConfig | None = None) -> AnalysisContext:
        config = config or ProcessingConfig()
        return AnalysisContext(assignment, repository, config, config_hash(config))

    return factory
