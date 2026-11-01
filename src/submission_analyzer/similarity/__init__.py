"""Code similarity (lab 2): token n-gram features, Jaccard and TF-IDF cosine, pair comparison."""

from submission_analyzer.similarity.features import CodeFeatures, extract_features
from submission_analyzer.similarity.index import FeatureIndex
from submission_analyzer.similarity.methods import (
    JaccardMethod,
    PairScore,
    ScoreStatus,
    SimilarityError,
    SimilarityMethod,
    TfidfCosineMethod,
    create_method,
)
from submission_analyzer.similarity.pairs import PairKind, PairResult, candidate_pairs, compare, compare_all, top_k

__all__ = [
    "CodeFeatures",
    "FeatureIndex",
    "JaccardMethod",
    "PairKind",
    "PairResult",
    "PairScore",
    "ScoreStatus",
    "SimilarityError",
    "SimilarityMethod",
    "TfidfCosineMethod",
    "candidate_pairs",
    "compare",
    "compare_all",
    "create_method",
    "extract_features",
    "top_k",
]
