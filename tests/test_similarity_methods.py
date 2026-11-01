"""One contract suite for both similarity methods, plus TF-IDF fit/transform discipline."""

from __future__ import annotations

import pytest

from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.ingestion.validation import validate_record
from submission_analyzer.similarity.features import CodeFeatures, extract_features
from submission_analyzer.similarity.methods import (
    JaccardMethod,
    ScoreStatus,
    SimilarityError,
    TfidfCosineMethod,
    create_method,
)

from .conftest import TEMPLATE

LOOP = (
    "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n    for i in range(1, n + 1):\n"
    "        if i % 15 == 0:\n            result.append('FizzBuzz')\n        elif i % 3 == 0:\n"
    "            result.append('Fizz')\n        else:\n            result.append(str(i))\n    return result\n"
)
REFORMATTED = "# my solution\n" + LOOP.replace("    ", "  ").replace("'Fizz')\n", "'Fizz')  # three\n")
RENAMED = LOOP.replace(" i ", " number ").replace("(i)", "(number)")  # the loop variable `i` -> `number`
CONCAT = (
    "def fizzbuzz(n: int) -> list[str]:\n    out = []\n    for k in range(1, n + 1):\n"
    "        word = ('Fizz' if k % 3 == 0 else '') + ('Buzz' if k % 5 == 0 else '')\n"
    "        out.append(word or str(k))\n    return out\n"
)
COMPREHENSION = (
    "def fizzbuzz(n: int) -> list[str]:\n"
    "    return ['FizzBuzz' if x % 15 == 0 else 'Fizz' if x % 3 == 0 else str(x) for x in range(1, n + 1)]\n"
)
EMPTY = "# nothing yet\n"


@pytest.fixture
def feats(make_record):
    counter = iter(range(1000))

    def factory(code: str, student: str | None = None, **config) -> CodeFeatures:
        no = next(counter)
        record = make_record(code=code, submission_id=f"s{no}", student_id=student or f"student-{no}")
        return extract_features(validate_record(record), TEMPLATE, SimilarityConfig(**config))

    return factory


@pytest.fixture(params=["jaccard", "tfidf"])
def method(request, feats):
    m = create_method(request.param)
    # Tuning set: other solutions of the assignment; the compared pairs come from elsewhere.
    m.fit([feats(LOOP), feats(CONCAT), feats(COMPREHENSION)])
    return m


def test_exact_copy_gets_maximal_non_empty_score(method, feats):
    result = method.score(feats(LOOP), feats(LOOP))
    assert (result.score, result.status) == (1.0, ScoreStatus.SCORED)


def test_reformatting_and_comments_do_not_lower_the_score(method, feats):
    assert method.score(feats(LOOP), feats(REFORMATTED)).score == 1.0


def test_renamed_variables_stay_highly_similar(method, feats):
    assert RENAMED != LOOP
    assert method.score(feats(LOOP), feats(RENAMED)).score >= 0.9


def test_independent_solutions_score_lower_than_derived(method, feats):
    independent = method.score(feats(LOOP), feats(CONCAT)).score
    derived = method.score(feats(LOOP), feats(RENAMED)).score
    assert independent is not None and derived is not None
    assert independent < 0.5 < derived


@pytest.mark.parametrize("other", [EMPTY, TEMPLATE])
def test_empty_features_give_null_score(method, feats, other):
    result = method.score(feats(EMPTY), feats(other))
    assert result.score is None
    assert result.status is ScoreStatus.INSUFFICIENT_DATA and result.reason


def test_broken_code_gives_null_score_with_reason(method, feats):
    result = method.score(feats("x = '''open\n"), feats("y = '''open\n"))
    assert result.score is None and "tokenized" in result.reason


@pytest.mark.parametrize(("a", "b"), [(LOOP, CONCAT), (LOOP, COMPREHENSION), (CONCAT, REFORMATTED), (LOOP, EMPTY)])
def test_score_is_symmetric_and_in_range(method, feats, a, b):
    fa, fb = feats(a), feats(b)
    ab, ba = method.score(fa, fb).score, method.score(fb, fa).score
    assert ab == ba
    assert ab is None or 0.0 <= ab <= 1.0


def test_jaccard_is_intersection_over_union(feats):
    a, b = feats(LOOP), feats(CONCAT)
    expected = len(a.grams.keys() & b.grams.keys()) / len(a.grams.keys() | b.grams.keys())
    assert JaccardMethod().score(a, b).score == pytest.approx(expected, abs=1e-6)


def test_without_normalization_renaming_lowers_jaccard(feats):
    plain = JaccardMethod().score(feats(LOOP, normalize_identifiers=False), feats(RENAMED, normalize_identifiers=False))
    assert plain.score is not None and plain.score < 1.0


# --- TF-IDF: fit only on the tuning set, transform everything else ---


def test_tfidf_requires_fit_before_scoring(feats):
    with pytest.raises(SimilarityError, match="fitted"):
        TfidfCosineMethod().score(feats(LOOP), feats(LOOP))


def test_tfidf_cannot_be_refitted(feats):
    method = TfidfCosineMethod()
    method.fit([feats(LOOP)])
    with pytest.raises(SimilarityError, match="already fitted"):
        method.fit([feats(CONCAT)])


def test_tfidf_scoring_does_not_change_vocabulary_or_idf(feats):
    method = TfidfCosineMethod()
    method.fit([feats(LOOP), feats(CONCAT)])
    vocabulary, idf = method.vocabulary, method.idf

    evaluation = feats(COMPREHENSION)
    method.score(evaluation, feats(REFORMATTED))

    assert method.vocabulary == vocabulary and method.idf == idf
    assert "for ID in" in vocabulary
    assert (")", "for", "ID") in evaluation.grams and ") for ID" not in vocabulary


def test_tfidf_ignores_ngrams_unseen_in_tuning(feats):
    method = TfidfCosineMethod()
    method.fit([feats(LOOP)])
    unseen = "while True:\n    pass\n"
    result = method.score(feats(unseen), feats(unseen))
    assert result.score is None and "tuning vocabulary" in result.reason


def test_tfidf_fit_needs_non_empty_tuning_set(feats):
    with pytest.raises(SimilarityError, match="at least one"):
        TfidfCosineMethod().fit([feats(EMPTY)])


def test_unknown_method_id():
    with pytest.raises(SimilarityError, match="unknown"):
        create_method("ast")
