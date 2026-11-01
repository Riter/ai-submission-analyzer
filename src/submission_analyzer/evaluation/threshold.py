"""Threshold baseline: ``review_required`` when one similarity score reaches a threshold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from submission_analyzer.evaluation.dataset import ExperimentError, PairRow
from submission_analyzer.evaluation.metrics import evaluate


class ThresholdDetector:
    """Picks the threshold with the best F1 on the tuning rows it is fitted on.

    Candidates are the lowest tuning score and the midpoints between
    neighbouring tuning scores; on equal F1 the higher threshold wins (fewer
    pairs sent to manual review). A missing score gives no decision.
    """

    detector_version = "1.0.0"

    def __init__(self, feature: str, detector_id: str | None = None) -> None:
        self.feature = feature
        self.detector_id = detector_id or f"threshold:{feature}"
        self.threshold: float | None = None

    def fit(self, rows: Sequence[PairRow]) -> None:
        scored = [(row.features[self.feature], row.label) for row in rows if row.features[self.feature] is not None]
        if not any(label for _, label in scored):
            raise ExperimentError(f"{self.detector_id}: no scored positive pairs in the tuning set")
        values = sorted({value for value, _ in scored})  # type: ignore[type-var]
        candidates = [values[0], *((lo + hi) / 2 for lo, hi in zip(values, values[1:], strict=False))]
        labels = [label for _, label in scored]

        def f1_at(threshold: float) -> float:
            predictions = [value >= threshold for value, _ in scored]  # type: ignore[operator]
            return evaluate(labels, predictions).matrix.f1 or 0.0

        self.threshold = round(max(candidates, key=lambda t: (f1_at(t), t)), 6)

    def predict(self, features: Mapping[str, float | None]) -> bool | None:
        if self.threshold is None:
            raise ExperimentError(f"{self.detector_id} must be fitted on the tuning set first")
        value = features[self.feature]
        return None if value is None else value >= self.threshold
