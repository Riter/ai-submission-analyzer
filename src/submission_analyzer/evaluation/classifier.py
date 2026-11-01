"""ML detector: a simple classifier of ``review_required`` on pair features.

The classifier output is treated as untrusted: probabilities are validated
before use. If they are invalid, the pair gets the threshold baseline's
decision and the result names the baseline, never the classifier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from submission_analyzer.evaluation.dataset import ExperimentError, PairRow
from submission_analyzer.evaluation.threshold import ThresholdDetector
from submission_analyzer.similarity.pair_features import PAIR_FEATURE_NAMES, PAIR_FEATURE_SCHEMA

DECISION_THRESHOLD = 0.5
# Features that can be missing (None): imputed with 0 plus an explicit "missing" indicator.
_OPTIONAL = ("jaccard", "tfidf_cosine", "template_shared", "size_ratio", "token_ratio")
VECTOR_NAMES: tuple[str, ...] = PAIR_FEATURE_NAMES + tuple(f"{name}_missing" for name in _OPTIONAL)


class FeatureSchemaError(Exception):
    """Pair features do not match the schema the classifier was built for."""


class ClassifierOutputError(Exception):
    """The classifier returned something that is not a valid probability."""


class ProbabilityEstimator(Protocol):
    def fit(self, x: Any, y: Any) -> Any: ...

    def predict_proba(self, x: Any) -> Any: ...


def vectorize(features: Mapping[str, float | None]) -> list[float]:
    if set(features) != set(PAIR_FEATURE_NAMES):
        raise FeatureSchemaError(
            f"pair features {sorted(features)} do not match schema {PAIR_FEATURE_SCHEMA} {sorted(PAIR_FEATURE_NAMES)}"
        )
    values = [float(features[name] or 0.0) for name in PAIR_FEATURE_NAMES]
    return values + [float(features[name] is None) for name in _OPTIONAL]


def default_estimator(seed: int) -> ProbabilityEstimator:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000),
    )


@dataclass(frozen=True, slots=True)
class Decision:
    review_required: bool | None
    detector_id: str
    probability: float | None = None
    fallback_reason: str | None = None


class PairClassifier:
    detector_id = "classifier:logistic_regression"
    detector_version = "1.0.0"

    def __init__(
        self,
        seed: int,
        feature_schema: str = PAIR_FEATURE_SCHEMA,
        estimator: ProbabilityEstimator | None = None,
    ) -> None:
        if feature_schema != PAIR_FEATURE_SCHEMA:
            raise FeatureSchemaError(
                f"classifier needs feature schema {PAIR_FEATURE_SCHEMA}, configuration requests {feature_schema}"
            )
        self.feature_schema = feature_schema
        self.estimator = estimator if estimator is not None else default_estimator(seed)
        self._fitted = False

    def fit(self, rows: Sequence[PairRow]) -> None:
        labels = [int(row.label) for row in rows]
        if len(set(labels)) < 2:
            raise ExperimentError("the classifier needs both derived and independent pairs in the tuning set")
        self.estimator.fit([vectorize(row.features) for row in rows], labels)
        self._fitted = True

    def check_schema(self, expected: str) -> None:
        """Refuse to apply a model trained on another feature schema."""
        if expected != self.feature_schema:
            raise FeatureSchemaError(f"model was trained on schema {self.feature_schema}, features are {expected}")

    def probabilities(self, rows: Sequence[PairRow]) -> list[float | None]:
        """Probability of the positive class per row; ``None`` marks an invalid value.

        Raises:
            ClassifierOutputError: the output as a whole is unusable (wrong shape).
        """
        if not self._fitted:
            raise ExperimentError("the classifier must be fitted on the tuning set first")
        if not rows:
            return []
        try:
            raw = self.estimator.predict_proba([vectorize(row.features) for row in rows])
            matrix = [list(map(float, line)) for line in raw]
        except (TypeError, ValueError) as exc:
            raise ClassifierOutputError(f"classifier output is not a probability matrix: {exc}") from exc
        if len(matrix) != len(rows) or any(len(line) != 2 for line in matrix):
            raise ClassifierOutputError(
                f"classifier returned shape {len(matrix)}x{len(matrix[0]) if matrix else 0}, expected {len(rows)}x2"
            )
        return [line[1] if all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in line) else None for line in matrix]


def classify_with_fallback(
    classifier: PairClassifier, fallback: ThresholdDetector, rows: Sequence[PairRow]
) -> list[Decision]:
    """Classifier decisions; rows with an invalid model output get the baseline's decision."""
    try:
        probabilities: list[float | None] = classifier.probabilities(rows)
        batch_error: str | None = None
    except ClassifierOutputError as exc:
        probabilities, batch_error = [None] * len(rows), str(exc)

    decisions: list[Decision] = []
    for row, probability in zip(rows, probabilities, strict=True):
        if row.undetermined:
            decisions.append(Decision(None, classifier.detector_id))
        elif probability is None:
            decisions.append(
                Decision(
                    fallback.predict(row.features),
                    fallback.detector_id,
                    fallback_reason=batch_error or "classifier returned an invalid probability",
                )
            )
        else:
            decisions.append(
                Decision(probability >= DECISION_THRESHOLD, classifier.detector_id, round(probability, 6))
            )
    return decisions
