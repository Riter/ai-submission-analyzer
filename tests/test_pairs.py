"""Which pairs are compared, top-k candidates and what a pair result explains."""

from __future__ import annotations

import pytest

from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.models import Submission
from submission_analyzer.core.textutils import source_lines
from submission_analyzer.ingestion.validation import validate_record
from submission_analyzer.similarity.features import extract_features
from submission_analyzer.similarity.index import FeatureIndex
from submission_analyzer.similarity.methods import JaccardMethod, ScoreStatus
from submission_analyzer.similarity.pairs import (
    TEMPLATE_ONLY_EXPLANATION,
    PairKind,
    candidate_pairs,
    compare,
    compare_all,
    iter_compare,
    top_k,
)

from .conftest import TEMPLATE
from .test_similarity_methods import COMPREHENSION, CONCAT, LOOP

OTHER_ASSIGNMENT = "lab-02-word-count"


@pytest.fixture
def corpus(make_record):
    """submission_id -> (Submission, template)."""
    rows = [
        ("a", "student-1", LOOP, "lab-01-fizzbuzz"),
        ("a-v2", "student-1", "# version 2\n" + LOOP, "lab-01-fizzbuzz"),  # same student, later version
        ("b", "student-2", LOOP, "lab-01-fizzbuzz"),  # exact copy of a
        ("c", "student-3", CONCAT, "lab-01-fizzbuzz"),
        ("d", "student-4", COMPREHENSION, "lab-01-fizzbuzz"),
        ("t1", "student-5", TEMPLATE, "lab-01-fizzbuzz"),  # only the template
        ("t2", "student-6", "# start\n" + TEMPLATE, "lab-01-fizzbuzz"),
        ("x", "student-7", LOOP, OTHER_ASSIGNMENT),  # same code, other assignment
    ]
    result: dict[str, tuple[Submission, str]] = {}
    for sid, student, code, assignment_id in rows:
        record = make_record(submission_id=sid, student_id=student, code=code, assignment_id=assignment_id)
        result[sid] = (validate_record(record), TEMPLATE if assignment_id == "lab-01-fizzbuzz" else "")
    return result


@pytest.fixture
def index(corpus):
    return FeatureIndex(extract_features(s, template, SimilarityConfig()) for s, template in corpus.values())


def ids(pairs) -> set[tuple[str, str]]:
    return {(a.submission_id, b.submission_id) for a, b in pairs}


def test_no_self_match_and_no_cross_assignment_pairs(index):
    candidates = candidate_pairs(index)
    for a, b in candidates.cross_student + candidates.same_student:
        assert a.submission_id != b.submission_id
        assert a.assignment_id == b.assignment_id
    assert not any("x" in pair for pair in ids(candidates.cross_student))


def test_versions_of_one_student_are_compared_separately(index):
    candidates = candidate_pairs(index)
    assert ids(candidates.same_student) == {("a", "a-v2")}
    assert ("a", "a-v2") not in ids(candidates.cross_student)
    _, same = compare_all(index, JaccardMethod())
    assert [(r.a, r.b, r.kind) for r in same] == [("a", "a-v2", PairKind.SAME_STUDENT)]


def test_exact_copy_tops_the_candidates(index):
    [best, *_] = top_k(index, "a", JaccardMethod(), k=3)
    assert (best.b, best.score) == ("b", 1.0)


def test_top_k_excludes_itself_own_versions_and_other_assignments(index):
    results = top_k(index, "a", JaccardMethod(), k=10)
    assert {r.b for r in results} == {"b", "c", "d", "t1", "t2"}
    assert len(top_k(index, "a", JaccardMethod(), k=2)) == 2


def test_top_k_is_sorted_and_puts_unscored_pairs_last(index):
    scores = [r.score for r in top_k(index, "a", JaccardMethod(), k=10)]
    assert scores == sorted(scores, reverse=True)
    # t1 and t2 contain only template lines: nothing to compare between them.
    last = top_k(index, "t1", JaccardMethod(), k=10)[-1]
    assert (last.b, last.score) == ("t2", None)


def test_top_k_unknown_submission(index):
    with pytest.raises(KeyError):
        top_k(index, "missing", JaccardMethod(), k=1)


def test_template_only_pair_is_explained_not_scored(index):
    result = compare(index.get("t1"), index.get("t2"), JaccardMethod())
    assert result.score is None and result.status is ScoreStatus.INSUFFICIENT_DATA
    assert result.template_shared == 1.0
    assert TEMPLATE_ONLY_EXPLANATION in result.explanation
    assert result.evidence_a == result.evidence_b == ()


def test_template_is_excluded_from_the_score(corpus):
    """Without the template both LOOP and COMPREHENSION look closer than they are."""
    (loop, _), (comp, _) = corpus["a"], corpus["d"]
    with_template = compare(
        extract_features(loop, TEMPLATE, SimilarityConfig()),
        extract_features(comp, TEMPLATE, SimilarityConfig()),
        JaccardMethod(),
    )
    blind = compare(
        extract_features(loop, "", SimilarityConfig()),
        extract_features(comp, "", SimilarityConfig()),
        JaccardMethod(),
    )
    assert with_template.score < blind.score
    assert with_template.template_shared and "template" in with_template.explanation
    assert "not a verdict" in with_template.explanation


def test_evidence_points_to_existing_lines_of_each_side(index, corpus):
    cross, same = compare_all(index, JaccardMethod())
    for result in cross + same:
        for sid, evidence in ((result.a, result.evidence_a), (result.b, result.evidence_b)):
            limit = len(source_lines(corpus[sid][0].code))
            assert all(1 <= e.start_line <= e.end_line <= limit for e in evidence), (sid, evidence)
    copy = compare(index.get("a"), index.get("b"), JaccardMethod())
    assert copy.evidence_a and copy.evidence_a == copy.evidence_b


def test_evidence_skips_comment_lines(index):
    result = compare(index.get("a-v2"), index.get("b"), JaccardMethod())
    assert all(e.start_line >= 2 for e in result.evidence_a)  # line 1 is "# version 2"


def test_syntax_error_is_reported_in_warnings(make_record):
    broken = validate_record(make_record(submission_id="s1", student_id="p1", code="def f(x)\n    return x\n"))
    fine = validate_record(make_record(submission_id="s2", student_id="p2", code="def f(x):\n    return x\n"))
    config = SimilarityConfig()
    result = compare(extract_features(broken, "", config), extract_features(fine, "", config), JaccardMethod())
    assert result.score is not None
    assert result.warnings and "s1 does not parse" in result.warnings[0]


def test_compare_rejects_self_and_cross_assignment(index):
    with pytest.raises(ValueError, match="itself"):
        compare(index.get("a"), index.get("a"), JaccardMethod())
    with pytest.raises(ValueError, match="same assignment"):
        compare(index.get("a"), index.get("x"), JaccardMethod())


def test_pair_result_serializes(index):
    data = compare(index.get("a"), index.get("c"), JaccardMethod()).to_dict()
    assert set(data) >= {"a", "b", "method_id", "method_version", "score", "status", "evidence_a", "evidence_b"}
    assert data["kind"] == "cross_student"


def test_threshold_keeps_scores_and_limits_evidence_to_review_pairs(index):
    """The optimization (evidence only where review is needed) must not change any score."""
    full = list(iter_compare(index, JaccardMethod()))
    flagged = list(iter_compare(index, JaccardMethod(), threshold=0.5))
    assert [(r.a, r.b, r.score, r.shared_grams) for r in full] == [(r.a, r.b, r.score, r.shared_grams) for r in flagged]
    for before, after in zip(full, flagged, strict=True):
        assert after.review_required == (None if after.score is None else after.score >= 0.5)
        if after.review_required:
            assert (after.evidence_a, after.evidence_b) == (before.evidence_a, before.evidence_b)
        else:
            assert after.evidence_a == after.evidence_b == ()
    assert any(r.review_required for r in flagged) and not all(r.review_required for r in flagged)
