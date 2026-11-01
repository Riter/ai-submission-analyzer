"""Pair comparison: which submissions are compared and what a result explains.

Only submissions of the same assignment by different students are candidates.
A submission is never compared with itself; versions of one student are
reported separately. A score is a signal for manual review, not a verdict.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from submission_analyzer.core.reports import Evidence, EvidenceSource
from submission_analyzer.core.textutils import line_runs
from submission_analyzer.similarity.features import CodeFeatures, Gram
from submission_analyzer.similarity.index import FeatureIndex
from submission_analyzer.similarity.methods import ScoreStatus, SimilarityMethod

SHARED_NOTE = "n-grams shared with the other submission (outside the allowed template)"
TEMPLATE_ONLY_EXPLANATION = (
    "Only the allowed assignment template is shared; using the template is permitted and is not evidence of copying."
)
SIGNAL_DISCLAIMER = "The score is a signal for manual review, not a verdict about a violation."


class PairKind(StrEnum):
    CROSS_STUDENT = "cross_student"
    SAME_STUDENT = "same_student"  # versions of one student's work


@dataclass(frozen=True, slots=True)
class PairResult:
    a: str
    b: str
    assignment_id: str
    kind: PairKind
    method_id: str
    method_version: str
    score: float | None
    status: ScoreStatus
    shared_grams: int
    template_shared: float | None
    explanation: str
    evidence_a: tuple[Evidence, ...] = ()
    evidence_b: tuple[Evidence, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "assignment_id": self.assignment_id,
            "kind": str(self.kind),
            "method_id": self.method_id,
            "method_version": self.method_version,
            "score": self.score,
            "status": str(self.status),
            "shared_grams": self.shared_grams,
            "template_shared": self.template_shared,
            "explanation": self.explanation,
            "evidence_a": [e.to_dict() for e in self.evidence_a],
            "evidence_b": [e.to_dict() for e in self.evidence_b],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CandidatePairs:
    cross_student: list[tuple[CodeFeatures, CodeFeatures]] = field(default_factory=list)
    same_student: list[tuple[CodeFeatures, CodeFeatures]] = field(default_factory=list)


def candidate_pairs(index: FeatureIndex) -> CandidatePairs:
    """All unordered pairs within each assignment, split by whether the student is the same."""
    result = CandidatePairs()
    for assignment_id in index.assignment_ids():
        items = index.for_assignment(assignment_id)
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                target = result.same_student if a.student_id == b.student_id else result.cross_student
                target.append((a, b))
    return result


def template_overlap(a: CodeFeatures, b: CodeFeatures) -> float | None:
    union = a.template_grams | b.template_grams
    if not union:
        return None
    return round(len(a.template_grams & b.template_grams) / len(union), 6)


def compare(a: CodeFeatures, b: CodeFeatures, method: SimilarityMethod) -> PairResult:
    if a.assignment_id != b.assignment_id:
        raise ValueError("only submissions of the same assignment are compared")
    if a.submission_id == b.submission_id:
        raise ValueError("a submission is not compared with itself")

    pair_score = method.score(a, b)
    shared = a.grams.keys() & b.grams.keys()
    template_shared = template_overlap(a, b)
    return PairResult(
        a=a.submission_id,
        b=b.submission_id,
        assignment_id=a.assignment_id,
        kind=PairKind.SAME_STUDENT if a.student_id == b.student_id else PairKind.CROSS_STUDENT,
        method_id=method.method_id,
        method_version=method.method_version,
        score=pair_score.score,
        status=pair_score.status,
        shared_grams=len(shared),
        template_shared=template_shared,
        explanation=_explain(pair_score.reason, len(shared), template_shared),
        evidence_a=_evidence(a, shared),
        evidence_b=_evidence(b, shared),
        warnings=tuple(
            f"{f.submission_id} does not parse ({f.syntax_error}); compared by tokens only"
            for f in (a, b)
            if f.syntax_error and not f.parse_error
        ),
    )


def _explain(reason: str | None, shared: int, template_shared: float | None) -> str:
    if reason is not None:
        text = f"Insufficient data: {reason}."
        if template_shared:
            text += " " + TEMPLATE_ONLY_EXPLANATION
        return text
    if shared == 0 and template_shared:
        return TEMPLATE_ONLY_EXPLANATION
    text = f"{shared} n-grams of the students' own code are shared."
    if template_shared:
        text += " Lines of the allowed template are excluded from the score."
    return f"{text} {SIGNAL_DISCLAIMER}"


def _evidence(features: CodeFeatures, shared: AbstractSet[Gram]) -> tuple[Evidence, ...]:
    lines = {
        line
        for gram in shared
        for start, end in features.grams[gram]
        for line in range(start, end + 1)
    }
    return tuple(Evidence(EvidenceSource.CODE, start, end, SHARED_NOTE) for start, end in line_runs(lines))


def compare_all(index: FeatureIndex, method: SimilarityMethod) -> tuple[list[PairResult], list[PairResult]]:
    """Compare every candidate pair; returns ``(cross_student, same_student)`` results."""
    candidates = candidate_pairs(index)
    return (
        [compare(a, b, method) for a, b in candidates.cross_student],
        [compare(a, b, method) for a, b in candidates.same_student],
    )


def top_k(index: FeatureIndex, submission_id: str, method: SimilarityMethod, k: int) -> list[PairResult]:
    """The ``k`` most similar submissions of the same assignment by other students.

    Excludes the submission itself, other versions by the same student and
    other assignments. Pairs without a score go last.
    """
    target = index.get(submission_id)
    if target is None:
        raise KeyError(f"submission '{submission_id}' is not indexed")
    if k < 1:
        raise ValueError("k must be positive")
    results = [
        compare(target, other, method)
        for other in index.for_assignment(target.assignment_id)
        if other.student_id != target.student_id
    ]
    results.sort(key=lambda r: (r.score is None, -(r.score or 0.0), r.b))
    return results[:k]
