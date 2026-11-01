"""Similarity methods behind one contract.

A score is a *signal* for manual review in ``[0, 1]``; ``None`` means there is
not enough data to compare (e.g. empty features), never "not similar".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from submission_analyzer.similarity.features import CodeFeatures


class SimilarityError(Exception):
    """A similarity method is used incorrectly (e.g. scored before fit)."""


class ScoreStatus(StrEnum):
    SCORED = "scored"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class PairScore:
    score: float | None
    status: ScoreStatus
    reason: str | None = None


class SimilarityMethod(Protocol):
    @property
    def method_id(self) -> str: ...

    @property
    def method_version(self) -> str: ...

    def fit(self, features: Iterable[CodeFeatures]) -> None:
        """Learn corpus statistics from the tuning set only."""

    def score(self, a: CodeFeatures, b: CodeFeatures) -> PairScore: ...


def _insufficient(a: CodeFeatures, b: CodeFeatures) -> PairScore:
    broken = [f"{f.submission_id}: {f.parse_error}" for f in (a, b) if f.parse_error]
    reason = (
        "code could not be tokenized (" + "; ".join(broken) + ")"
        if broken
        else "no n-grams outside the allowed template in both submissions"
    )
    return PairScore(None, ScoreStatus.INSUFFICIENT_DATA, reason)


def _round(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


class JaccardMethod:
    """Token baseline: |A ∩ B| / |A ∪ B| over sets of the students' own n-grams."""

    method_id = "jaccard"
    method_version = "1.0.0"

    def fit(self, features: Iterable[CodeFeatures]) -> None:
        """Jaccard has no corpus statistics; accepted for a uniform contract."""

    def score(self, a: CodeFeatures, b: CodeFeatures) -> PairScore:
        union = a.grams.keys() | b.grams.keys()
        if not union:
            return _insufficient(a, b)
        shared = a.grams.keys() & b.grams.keys()
        return PairScore(_round(len(shared) / len(union)), ScoreStatus.SCORED)


class TfidfCosineMethod:
    """Cosine similarity of TF-IDF vectors over the students' own n-grams.

    The vocabulary and IDF weights come only from the tuning set passed to
    :meth:`fit`; any other submission is only transformed. N-grams that never
    occur in the tuning set therefore carry no weight.
    """

    method_id = "tfidf"
    method_version = "1.0.0"

    def __init__(self) -> None:
        # Imported lazily: scikit-learn is only needed when this method is used.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(analyzer=_gram_terms, lowercase=False)
        self._fitted = False
        self._cache: dict[str, object] = {}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def vocabulary(self) -> dict[str, int]:
        self._require_fit()
        return dict(self._vectorizer.vocabulary_)

    @property
    def idf(self) -> tuple[float, ...]:
        self._require_fit()
        return tuple(float(x) for x in self._vectorizer.idf_)

    def fit(self, features: Iterable[CodeFeatures]) -> None:
        if self._fitted:
            raise SimilarityError("tfidf is already fitted; create a new method to refit on another tuning set")
        docs = [f for f in features if not f.is_empty]
        if not docs:
            raise SimilarityError("tfidf needs at least one non-empty tuning submission to fit")
        self._vectorizer.fit(docs)
        self._fitted = True

    def score(self, a: CodeFeatures, b: CodeFeatures) -> PairScore:
        self._require_fit()
        if a.is_empty and b.is_empty:
            return _insufficient(a, b)
        va, vb = self._vector(a), self._vector(b)
        if va.nnz == 0 or vb.nnz == 0:
            return PairScore(
                None,
                ScoreStatus.INSUFFICIENT_DATA,
                "no n-grams from the tuning vocabulary in at least one submission",
            )
        # Rows are L2-normalized by TfidfVectorizer, so the dot product is the cosine.
        return PairScore(_round(float(va.multiply(vb).sum())), ScoreStatus.SCORED)

    def _vector(self, features: CodeFeatures):  # -> scipy.sparse row
        key = features.content_hash
        if key not in self._cache:
            self._cache[key] = self._vectorizer.transform([features])
        return self._cache[key]

    def _require_fit(self) -> None:
        if not self._fitted:
            raise SimilarityError("tfidf must be fitted on the tuning set before scoring")


def _gram_terms(features: CodeFeatures) -> list[str]:
    """One term per occurrence of an own n-gram (term frequency matters for TF-IDF)."""
    return [" ".join(gram) for gram, spans in features.grams.items() for _ in spans]


_METHODS: dict[str, type[JaccardMethod] | type[TfidfCosineMethod]] = {
    JaccardMethod.method_id: JaccardMethod,
    TfidfCosineMethod.method_id: TfidfCosineMethod,
}


def create_method(method_id: str) -> SimilarityMethod:
    try:
        return _METHODS[method_id]()
    except KeyError:
        raise SimilarityError(f"unknown similarity method '{method_id}'") from None
