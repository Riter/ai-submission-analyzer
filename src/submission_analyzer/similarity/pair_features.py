"""Pair features for the ML track: the input of the ``review_required`` classifier (lab 2, stage B).

The feature schema is versioned. A model trained on one schema must not be
applied to features of another (``processing.feature_schema_version``).
"""

from __future__ import annotations

from collections.abc import Sequence

from submission_analyzer.similarity.features import CodeFeatures
from submission_analyzer.similarity.methods import SimilarityMethod
from submission_analyzer.similarity.pairs import template_overlap

PAIR_FEATURE_SCHEMA = "pair-1"

# Fixed order: classifiers consume a vector, not a dict.
PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "jaccard",
    "tfidf_cosine",
    "template_shared",
    "shared_grams",
    "size_ratio",
    "token_ratio",
    "has_parse_error",
    "has_syntax_error",
)


def pair_features(a: CodeFeatures, b: CodeFeatures, methods: Sequence[SimilarityMethod]) -> dict[str, float | None]:
    """Features of one pair. A missing score stays ``None``; the classifier decides how to impute it."""
    scores = {m.method_id: m.score(a, b).score for m in methods}
    size_a, size_b = len(a.grams), len(b.grams)
    return {
        "jaccard": scores.get("jaccard"),
        "tfidf_cosine": scores.get("tfidf"),
        "template_shared": template_overlap(a, b),
        "shared_grams": float(len(a.grams.keys() & b.grams.keys())),
        "size_ratio": _ratio(size_a, size_b),
        "token_ratio": _ratio(a.token_count, b.token_count),
        "has_parse_error": float(bool(a.parse_error or b.parse_error)),
        "has_syntax_error": float(bool(a.syntax_error or b.syntax_error)),
    }


def _ratio(x: int, y: int) -> float | None:
    """min/max in [0, 1]; ``None`` when both are zero."""
    if max(x, y) == 0:
        return None
    return round(min(x, y) / max(x, y), 6)
