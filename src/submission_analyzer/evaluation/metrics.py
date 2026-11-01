"""Quality metrics for the ``review_required`` signal.

Positive class: the pair is derived from one work (``derivation_relation`` is
not ``independent``). A pair without a decision (no score) is *undetermined*:
it is kept out of the confusion matrix and reported as a separate share.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None if p is None or r is None else 0.0
        return round(2 * p * r / (p + r), 6)

    def to_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass(frozen=True, slots=True)
class Evaluation:
    matrix: ConfusionMatrix
    undetermined: int
    total: int

    @property
    def undetermined_share(self) -> float | None:
        return _ratio(self.undetermined, self.total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusion_matrix": self.matrix.to_dict(),
            "precision": self.matrix.precision,
            "recall": self.matrix.recall,
            "f1": self.matrix.f1,
            "undetermined": self.undetermined,
            "undetermined_share": self.undetermined_share,
            "total": self.total,
        }


def evaluate(labels: Sequence[bool], predictions: Sequence[bool | None]) -> Evaluation:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    tp = fp = fn = tn = undetermined = 0
    for label, predicted in zip(labels, predictions, strict=True):
        if predicted is None:
            undetermined += 1
        elif predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    return Evaluation(ConfusionMatrix(tp, fp, fn, tn), undetermined, len(labels))


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None
