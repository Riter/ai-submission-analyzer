"""Threshold baseline: chosen on the tuning set only, deterministically."""

from __future__ import annotations

import pytest

from submission_analyzer.core.corpus import SplitName
from submission_analyzer.evaluation.dataset import ExperimentError
from submission_analyzer.evaluation.threshold import ThresholdDetector

from .evaluation_helpers import row, separable


def test_threshold_is_the_midpoint_of_the_gap():
    detector = ThresholdDetector("jaccard")
    detector.fit(separable())
    assert detector.threshold == pytest.approx((0.45 + 0.7) / 2)
    assert detector.predict(row(0.6, True).features) is True
    assert detector.predict(row(0.5, False).features) is False


def test_threshold_depends_only_on_the_rows_it_was_fitted_on():
    tune = separable()
    first, second = ThresholdDetector("jaccard"), ThresholdDetector("jaccard")
    first.fit(tune)
    second.fit(tune)
    # Evaluation rows with arbitrary labels are only predicted, never fitted on.
    for r in [row(0.95, False, split=SplitName.EVAL), row(0.05, True, split=SplitName.EVAL)]:
        second.predict(r.features)
    assert first.threshold == second.threshold


def test_tie_goes_to_the_higher_threshold():
    # Thresholds 0.2 and 0.8 both give F1 = 2/3 here; the higher one sends fewer pairs to review.
    labels = {0.1: False, 0.3: True, 0.5: False, 0.7: False, 0.9: True}
    detector = ThresholdDetector("jaccard")
    detector.fit([row(score, derived) for score, derived in labels.items()])
    assert detector.threshold == pytest.approx(0.8)


def test_missing_score_gives_no_decision():
    detector = ThresholdDetector("jaccard")
    detector.fit(separable())
    assert detector.predict(row(None, True).features) is None


def test_errors():
    with pytest.raises(ExperimentError, match="fitted"):
        ThresholdDetector("jaccard").predict(row(0.5, True).features)
    with pytest.raises(ExperimentError, match="no scored positive"):
        ThresholdDetector("jaccard").fit([row(0.2, False), row(None, True)])
