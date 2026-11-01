"""Reproducible comparison of ``review_required`` detectors on a fixed split.

Every detector is fitted on the tune side only (thresholds, TF-IDF vocabulary,
classifier weights) and evaluated on the eval side. Scores and decisions are
signals for manual review; they do not establish a violation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from submission_analyzer.core.config import ProcessingConfig, config_hash
from submission_analyzer.core.corpus import LabeledPair, Split, SplitName, check_no_family_leakage
from submission_analyzer.evaluation.classifier import Decision, PairClassifier, classify_with_fallback
from submission_analyzer.evaluation.dataset import ExperimentError, PairRow, build_pair_rows, by_split, fit_methods
from submission_analyzer.evaluation.metrics import Evaluation, evaluate
from submission_analyzer.evaluation.threshold import ThresholdDetector
from submission_analyzer.similarity.index import FeatureIndex
from submission_analyzer.similarity.pair_features import PAIR_FEATURE_SCHEMA

TEMPLATE_BLIND_ID = "threshold:jaccard_template_blind"


@dataclass(frozen=True, slots=True)
class DetectorReport:
    detector_id: str
    detector_version: str
    threshold: float | None
    tune: Evaluation
    eval: Evaluation
    fallbacks: int = 0
    # Highest score among eval pairs that share only the allowed template: how close they come to the threshold.
    template_only_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "detector_version": self.detector_version,
            "threshold": self.threshold,
            "tune": self.tune.to_dict(),
            "eval": self.eval.to_dict(),
            "fallbacks": self.fallbacks,
            "template_only_max": self.template_only_max,
        }


@dataclass(frozen=True, slots=True)
class ErrorCase:
    detector_id: str
    kind: str  # "false_positive" | "false_negative"
    pair: LabeledPair
    features: Mapping[str, float | None]
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "kind": self.kind,
            "a": self.pair.a,
            "b": self.pair.b,
            "assignment_id": self.pair.assignment_id,
            "derivation_relation": str(self.pair.derivation_relation),
            "template_only": self.pair.template_only,
            "features": dict(self.features),
            "probability": self.probability,
        }


@dataclass(slots=True)
class ExperimentResult:
    params: dict[str, Any]
    detectors: list[DetectorReport]
    errors: list[ErrorCase]
    classifier: PairClassifier
    decisions: dict[str, list[Decision]] = field(default_factory=dict)  # eval decisions per detector

    def detector(self, detector_id: str) -> DetectorReport:
        return next(d for d in self.detectors if d.detector_id == detector_id)

    def metrics(self) -> dict[str, float]:
        """Flat numeric metrics, e.g. ``eval/threshold:jaccard/f1``."""
        flat: dict[str, float] = {}
        for report in self.detectors:
            for side, evaluation in (("tune", report.tune), ("eval", report.eval)):
                for name in ("precision", "recall", "f1", "undetermined_share"):
                    value = evaluation.to_dict()[name]
                    if value is not None:
                        flat[f"{side}/{report.detector_id}/{name}"] = value
            if report.threshold is not None:
                flat[f"threshold/{report.detector_id}"] = report.threshold
            flat[f"fallbacks/{report.detector_id}"] = float(report.fallbacks)
        return flat

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "detectors": [d.to_dict() for d in self.detectors],
            "errors": [e.to_dict() for e in self.errors],
        }


def run_experiment(
    index: FeatureIndex,
    pairs: Sequence[LabeledPair],
    split: Split,
    config: ProcessingConfig,
    *,
    template_blind_index: FeatureIndex | None = None,
    inputs: Mapping[str, str] | None = None,
) -> ExperimentResult:
    """Fit every detector on the tune side and evaluate it on the eval side.

    ``template_blind_index`` (features built without the allowed templates) adds
    an ablation detector that shows what the template exclusion is worth.
    ``inputs`` (e.g. file hashes) is recorded in the parameters.
    """
    problems = check_no_family_leakage(split, pairs)
    if problems:
        raise ExperimentError("family leakage between tune and eval: " + "; ".join(problems[:3]))
    if config.feature_schema_version != PAIR_FEATURE_SCHEMA:
        raise ExperimentError(
            f"processing.feature_schema_version is {config.feature_schema_version}, "
            f"the pair features implement {PAIR_FEATURE_SCHEMA}"
        )

    rows = build_pair_rows(index, pairs, fit_methods(index, split, config.similarity))
    tune, evaluation = by_split(rows, SplitName.TUNE), by_split(rows, SplitName.EVAL)
    if not tune or not evaluation:
        raise ExperimentError("both the tune and the eval side need labeled pairs")

    reports: list[DetectorReport] = []
    errors: list[ErrorCase] = []
    decisions: dict[str, list[Decision]] = {}

    def add_threshold(detector: ThresholdDetector, tune_rows: list[PairRow], eval_rows: list[PairRow]) -> None:
        detector.fit(tune_rows)
        eval_decisions = [Decision(detector.predict(r.features), detector.detector_id) for r in eval_rows]
        reports.append(
            DetectorReport(
                detector.detector_id,
                detector.detector_version,
                detector.threshold,
                evaluate([r.label for r in tune_rows], [detector.predict(r.features) for r in tune_rows]),
                _evaluate(eval_rows, eval_decisions),
                template_only_max=max(
                    (
                        value
                        for r in eval_rows
                        if r.pair.template_only and (value := r.features[detector.feature]) is not None
                    ),
                    default=None,
                ),
            )
        )
        decisions[detector.detector_id] = eval_decisions
        errors.extend(_errors(detector.detector_id, eval_rows, eval_decisions))

    jaccard = ThresholdDetector("jaccard")
    add_threshold(jaccard, tune, evaluation)
    if "tfidf" in config.similarity.methods:
        add_threshold(ThresholdDetector("tfidf_cosine"), tune, evaluation)

    classifier = PairClassifier(config.experiment.seed, config.feature_schema_version or "")
    classifier.fit(tune)
    tune_decisions = classify_with_fallback(classifier, jaccard, tune)
    eval_decisions = classify_with_fallback(classifier, jaccard, evaluation)
    reports.append(
        DetectorReport(
            classifier.detector_id,
            classifier.detector_version,
            None,
            _evaluate(tune, tune_decisions),
            _evaluate(evaluation, eval_decisions),
            fallbacks=sum(d.fallback_reason is not None for d in eval_decisions),
            template_only_max=max(
                (d.probability for r, d in zip(evaluation, eval_decisions, strict=True)
                 if r.pair.template_only and d.probability is not None),
                default=None,
            ),  # fmt: skip
        )
    )
    decisions[classifier.detector_id] = eval_decisions
    errors.extend(_errors(classifier.detector_id, evaluation, eval_decisions))

    if template_blind_index is not None:
        blind_rows = build_pair_rows(
            template_blind_index, pairs, fit_methods(template_blind_index, split, config.similarity)
        )
        add_threshold(
            ThresholdDetector("jaccard", TEMPLATE_BLIND_ID),
            by_split(blind_rows, SplitName.TUNE),
            by_split(blind_rows, SplitName.EVAL),
        )

    params = {
        "config_hash": config_hash(config),
        "seed": config.experiment.seed,
        "classifier": config.experiment.classifier,
        "feature_schema_version": config.feature_schema_version,
        "ngram": config.similarity.ngram,
        "normalize_identifiers": config.similarity.normalize_identifiers,
        "methods": ",".join(config.similarity.methods),
        "decision_threshold": 0.5,
        "pairs_tune": len(tune),
        "pairs_eval": len(evaluation),
        "positives_tune": sum(r.label for r in tune),
        "positives_eval": sum(r.label for r in evaluation),
        **dict(inputs or {}),
    }
    return ExperimentResult(params, reports, errors, classifier, decisions)


def _evaluate(rows: Iterable[PairRow], decisions: Sequence[Decision]) -> Evaluation:
    return evaluate([r.label for r in rows], [d.review_required for d in decisions])


def _errors(detector_id: str, rows: Sequence[PairRow], decisions: Sequence[Decision]) -> list[ErrorCase]:
    cases: list[ErrorCase] = []
    for row, decision in zip(rows, decisions, strict=True):
        if decision.review_required is None or decision.review_required == row.label:
            continue
        kind = "false_positive" if decision.review_required else "false_negative"
        cases.append(ErrorCase(detector_id, kind, row.pair, row.features, decision.probability))
    return cases
