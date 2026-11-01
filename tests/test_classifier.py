"""Classifier on pair features: schema checks and invalid model output."""

from __future__ import annotations

import pytest

from submission_analyzer.core.corpus import SplitName
from submission_analyzer.evaluation.classifier import (
    VECTOR_NAMES,
    ClassifierOutputError,
    FeatureSchemaError,
    PairClassifier,
    classify_with_fallback,
    vectorize,
)
from submission_analyzer.evaluation.dataset import ExperimentError
from submission_analyzer.evaluation.threshold import ThresholdDetector

from .evaluation_helpers import row, separable


class FakeEstimator:
    """Returns a fixed output from predict_proba, whatever it is."""

    def __init__(self, output):
        self.output = output

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        return self.output


@pytest.fixture
def fallback():
    detector = ThresholdDetector("jaccard")
    detector.fit(separable())
    return detector


def fitted(estimator=None) -> PairClassifier:
    classifier = PairClassifier(seed=42, estimator=estimator)
    classifier.fit(separable())
    return classifier


def test_classifier_learns_the_separable_case(fallback):
    rows = [row(0.95, True, split=SplitName.EVAL), row(0.15, False, split=SplitName.EVAL)]
    decisions = classify_with_fallback(fitted(), fallback, rows)
    assert [d.review_required for d in decisions] == [True, False]
    assert all(d.detector_id == PairClassifier.detector_id and d.fallback_reason is None for d in decisions)
    assert all(0.0 <= d.probability <= 1.0 for d in decisions)


@pytest.mark.parametrize(
    "bad_row",
    [[float("nan"), float("nan")], [-0.5, 1.5], [0.2, float("inf")]],
    ids=["nan", "out_of_range", "inf"],
)
def test_invalid_probability_falls_back_to_the_baseline(fallback, bad_row):
    rows = [row(0.95, True), row(0.15, False)]
    classifier = fitted(FakeEstimator([[0.1, 0.9], bad_row]))

    good, bad = classify_with_fallback(classifier, fallback, rows)

    assert (good.detector_id, good.review_required, good.fallback_reason) == (PairClassifier.detector_id, True, None)
    # The fallback names the baseline that decided — it is not presented as the model's result.
    assert bad.detector_id == "threshold:jaccard" and bad.probability is None
    assert bad.review_required is False and "invalid probability" in bad.fallback_reason


@pytest.mark.parametrize(
    "output",
    [[[0.1, 0.9]], [[0.9], [0.1]], "not a matrix", [[0.5, "x"], [0.5, 0.5]]],
    ids=["too_few_rows", "one_column", "string", "non_numeric"],
)
def test_malformed_output_falls_back_for_the_whole_batch(fallback, output):
    rows = [row(0.95, True), row(0.15, False)]
    with pytest.raises(ClassifierOutputError):
        fitted(FakeEstimator(output)).probabilities(rows)
    decisions = classify_with_fallback(fitted(FakeEstimator(output)), fallback, rows)
    assert [d.detector_id for d in decisions] == ["threshold:jaccard"] * 2
    assert [d.review_required for d in decisions] == [True, False]


def test_undetermined_pair_gets_no_decision(fallback):
    [decision] = classify_with_fallback(fitted(), fallback, [row(None, True, tfidf=None)])
    assert decision.review_required is None


def test_incompatible_feature_schema_is_rejected():
    with pytest.raises(FeatureSchemaError, match="pair-1"):
        PairClassifier(seed=42, feature_schema="pair-0")
    classifier = fitted()
    with pytest.raises(FeatureSchemaError, match="trained on schema pair-1"):
        classifier.check_schema("pair-2")
    features = dict(row(0.5, True).features)
    features["extra_feature"] = 1.0
    with pytest.raises(FeatureSchemaError):
        vectorize(features)


def test_vector_has_explicit_missing_indicators():
    vector = vectorize(row(None, True, tfidf=0.4).features)
    assert len(vector) == len(VECTOR_NAMES)
    assert vector[VECTOR_NAMES.index("jaccard")] == 0.0
    assert vector[VECTOR_NAMES.index("jaccard_missing")] == 1.0
    assert vector[VECTOR_NAMES.index("tfidf_cosine_missing")] == 0.0


def test_classifier_needs_both_classes_and_fit():
    with pytest.raises(ExperimentError, match="both"):
        PairClassifier(seed=1).fit([row(0.9, True), row(0.8, True)])
    with pytest.raises(ExperimentError, match="fitted"):
        PairClassifier(seed=1).probabilities([row(0.9, True)])
