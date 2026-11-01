"""Builders for labeled pair rows used by the evaluation tests."""

from __future__ import annotations

from itertools import count

from submission_analyzer.core.corpus import DerivationRelation, LabeledPair, SplitName
from submission_analyzer.evaluation.dataset import PairRow
from submission_analyzer.similarity.pair_features import PAIR_FEATURE_NAMES

_ids = count()


def row(
    jaccard: float | None,
    derived: bool,
    *,
    split: SplitName = SplitName.TUNE,
    tfidf: float | None = None,
    template_only: bool = False,
) -> PairRow:
    no = next(_ids)
    relation = DerivationRelation.MODIFIED if derived else DerivationRelation.INDEPENDENT
    pair = LabeledPair(f"a{no}", f"b{no}", "lab", f"fa{no}", f"fb{no}", relation, template_only, split)
    features: dict[str, float | None] = dict.fromkeys(PAIR_FEATURE_NAMES, 0.0)
    features.update(
        jaccard=jaccard,
        tfidf_cosine=jaccard if tfidf is None else tfidf,
        template_shared=1.0 if template_only else 0.5,
        shared_grams=(jaccard or 0.0) * 50,
        size_ratio=0.9,
        token_ratio=0.9,
    )
    return PairRow(pair, features)


def separable(split: SplitName = SplitName.TUNE) -> list[PairRow]:
    """Derived pairs score high, independent ones low."""
    return [row(s, True, split=split) for s in (0.7, 0.8, 0.9, 1.0)] + [
        row(s, False, split=split) for s in (0.1, 0.2, 0.3, 0.45)
    ]
