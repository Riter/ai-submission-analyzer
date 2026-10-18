"""Submission storage.

The service depends on the :class:`SubmissionRepository` protocol, not on a
concrete class. Lab 1 only needs the in-memory implementation; a persistent
one (SQLite) is added in lab 3.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from submission_analyzer.models import Submission


class SubmissionRepository(Protocol):
    def get(self, submission_id: str) -> Submission | None: ...

    def add(self, submission: Submission) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Submission]: ...


class InMemorySubmissionRepository:
    def __init__(self) -> None:
        self._items: dict[str, Submission] = {}

    def get(self, submission_id: str) -> Submission | None:
        return self._items.get(submission_id)

    def add(self, submission: Submission) -> None:
        if submission.submission_id in self._items:
            # The service resolves duplicates/conflicts before calling add();
            # reaching this line means a caller bypassed it.
            raise KeyError(f"submission '{submission.submission_id}' already stored")
        self._items[submission.submission_id] = submission

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Submission]:
        return iter(self._items.values())
