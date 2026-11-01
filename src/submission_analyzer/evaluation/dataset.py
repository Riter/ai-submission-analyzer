"""Labeled pair dataset: corpus labels joined with pair features.

Similarity methods with corpus statistics (TF-IDF) are fitted on the tune side
only, so evaluation pairs never influence the features they are scored with.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.corpus import LabeledPair, Split, SplitName
from submission_analyzer.similarity.index import FeatureIndex
from submission_analyzer.similarity.methods import SimilarityMethod, create_method
from submission_analyzer.similarity.pair_features import pair_features


class ExperimentError(Exception):
    """The experiment cannot be run on this data (leakage, missing submissions, one class only)."""


@dataclass(frozen=True, slots=True)
class PairRow:
    pair: LabeledPair
    features: Mapping[str, float | None]

    @property
    def label(self) -> bool:
        """Positive class: the pair is derived from one work."""
        return self.pair.is_derived

    @property
    def undetermined(self) -> bool:
        """No similarity method produced a score for this pair."""
        return self.features["jaccard"] is None and self.features["tfidf_cosine"] is None


def fit_methods(index: FeatureIndex, split: Split, config: SimilarityConfig) -> list[SimilarityMethod]:
    tune_ids = split.submission_ids(SplitName.TUNE)
    methods = [create_method(method_id) for method_id in config.methods]
    for method in methods:
        method.fit(f for f in index if f.submission_id in tune_ids)
    return methods


def build_pair_rows(
    index: FeatureIndex, pairs: Iterable[LabeledPair], methods: Sequence[SimilarityMethod]
) -> list[PairRow]:
    rows: list[PairRow] = []
    for pair in pairs:
        a, b = index.get(pair.a), index.get(pair.b)
        if a is None or b is None:
            missing = pair.a if a is None else pair.b
            raise ExperimentError(f"labeled pair {pair.a}/{pair.b}: submission {missing} is not in the corpus")
        rows.append(PairRow(pair, pair_features(a, b, methods)))
    return rows


def by_split(rows: Iterable[PairRow], side: SplitName) -> list[PairRow]:
    return [row for row in rows if row.pair.split is side]
