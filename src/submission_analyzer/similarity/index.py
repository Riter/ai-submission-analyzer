"""In-memory feature index.

The index keeps features (n-grams with line spans), not raw submission text.
Its size still grows with the corpus — O(N · G) for N submissions with about
G distinct n-grams each — so streaming input does not make it constant-memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from submission_analyzer.similarity.features import CodeFeatures


class FeatureIndex:
    def __init__(self, features: Iterable[CodeFeatures] = ()) -> None:
        self._by_id: dict[str, CodeFeatures] = {}
        self._by_assignment: dict[str, list[str]] = {}
        for item in features:
            self.add(item)

    def add(self, features: CodeFeatures) -> None:
        if features.submission_id in self._by_id:
            raise ValueError(f"submission '{features.submission_id}' is already indexed")
        self._by_id[features.submission_id] = features
        self._by_assignment.setdefault(features.assignment_id, []).append(features.submission_id)

    def get(self, submission_id: str) -> CodeFeatures | None:
        return self._by_id.get(submission_id)

    def for_assignment(self, assignment_id: str) -> list[CodeFeatures]:
        return [self._by_id[sid] for sid in self._by_assignment.get(assignment_id, ())]

    def assignment_ids(self) -> list[str]:
        return list(self._by_assignment)

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[CodeFeatures]:
        return iter(self._by_id.values())

    def __contains__(self, submission_id: object) -> bool:
        return submission_id in self._by_id
