"""Confusion matrix, precision/recall/F1 and the share of undetermined results."""

from __future__ import annotations

import pytest

from submission_analyzer.evaluation.metrics import ConfusionMatrix, evaluate


def test_hand_computed_example():
    labels = [True, True, True, False, False, False, True, False]
    predictions = [True, True, False, True, False, False, None, None]

    result = evaluate(labels, predictions)

    assert result.matrix == ConfusionMatrix(tp=2, fp=1, fn=1, tn=2)
    assert result.matrix.precision == pytest.approx(2 / 3, abs=1e-6)
    assert result.matrix.recall == pytest.approx(2 / 3, abs=1e-6)
    assert result.matrix.f1 == pytest.approx(2 / 3, abs=1e-6)
    assert (result.undetermined, result.total, result.undetermined_share) == (2, 8, 0.25)


def test_undetermined_pairs_are_not_counted_as_errors():
    result = evaluate([True, False], [None, None])
    assert result.matrix == ConfusionMatrix()
    assert result.undetermined_share == 1.0


@pytest.mark.parametrize(
    ("matrix", "precision", "recall", "f1"),
    [
        (ConfusionMatrix(tn=5), None, None, None),  # nothing predicted, nothing to find
        (ConfusionMatrix(fn=3, tn=5), None, 0.0, None),  # nothing predicted positive
        (ConfusionMatrix(fp=2), 0.0, None, None),
        (ConfusionMatrix(tp=0, fp=1, fn=1), 0.0, 0.0, 0.0),
        (ConfusionMatrix(tp=4), 1.0, 1.0, 1.0),
    ],
)
def test_division_by_zero_gives_none(matrix, precision, recall, f1):
    assert (matrix.precision, matrix.recall, matrix.f1) == (precision, recall, f1)


def test_lengths_must_match():
    with pytest.raises(ValueError):
        evaluate([True], [])


def test_serialization():
    data = evaluate([True, False], [True, None]).to_dict()
    assert data["confusion_matrix"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}
    assert data["undetermined_share"] == 0.5
