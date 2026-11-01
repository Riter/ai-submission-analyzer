"""Evaluation of the review_required signal (lab 2, stage B): metrics, threshold baseline, classifier."""

from submission_analyzer.evaluation.classifier import (
    ClassifierOutputError,
    Decision,
    FeatureSchemaError,
    PairClassifier,
    classify_with_fallback,
)
from submission_analyzer.evaluation.dataset import ExperimentError, PairRow, build_pair_rows, fit_methods
from submission_analyzer.evaluation.experiment import ExperimentResult, run_experiment
from submission_analyzer.evaluation.metrics import ConfusionMatrix, Evaluation, evaluate
from submission_analyzer.evaluation.threshold import ThresholdDetector

__all__ = [
    "ClassifierOutputError",
    "ConfusionMatrix",
    "Decision",
    "Evaluation",
    "ExperimentError",
    "ExperimentResult",
    "FeatureSchemaError",
    "PairClassifier",
    "PairRow",
    "ThresholdDetector",
    "build_pair_rows",
    "classify_with_fallback",
    "evaluate",
    "fit_methods",
    "run_experiment",
]
