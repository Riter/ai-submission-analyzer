"""Local, deterministic requirements analyzer (no model).

Each assignment requirement becomes a criterion. Checkable markers are
extracted from the requirement wording — function names (``word_count(``) and
quoted literals (``'Fizz'``) — and searched for in the submission. When a
requirement has no such markers the analyzer honestly reports
``insufficient_evidence`` instead of guessing.
"""

from __future__ import annotations

import re

from submission_analyzer.analyzers.base import AnalysisContext, build_report
from submission_analyzer.core.models import Submission
from submission_analyzer.core.reports import (
    AnalysisReport,
    CriterionStatus,
    Evidence,
    EvidenceSource,
    RubricFeedback,
)
from submission_analyzer.core.textutils import source_lines

_FUNCTION_RE = re.compile(r"\b([A-Za-z_]\w*)\(")
_QUOTED_RE = re.compile(r"'([^']+)'")


def requirement_markers(requirement: str) -> tuple[str, ...]:
    markers = [*_FUNCTION_RE.findall(requirement), *_QUOTED_RE.findall(requirement)]
    return tuple(dict.fromkeys(markers))  # de-duplicate, keep order


class RequirementsAnalyzer:
    analyzer_id = "requirements"
    analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        sources = {
            EvidenceSource.CODE: source_lines(submission.code),
            EvidenceSource.TEXT: source_lines(submission.text),
        }
        feedback = tuple(
            self._check(f"req-{i}", requirement, sources)
            for i, requirement in enumerate(context.assignment.requirements, start=1)
        )
        return build_report(self, submission, context, rubric_feedback=feedback)

    def _check(
        self,
        criterion_id: str,
        requirement: str,
        sources: dict[EvidenceSource, list[str]],
    ) -> RubricFeedback:
        markers = requirement_markers(requirement)
        if not markers:
            return RubricFeedback(
                criterion_id,
                CriterionStatus.INSUFFICIENT_EVIDENCE,
                f"'{requirement}': no checkable markers; needs model or manual review.",
            )

        evidence: list[Evidence] = []
        missing: list[str] = []
        for marker in markers:
            location = _find(marker, sources)
            if location is None:
                missing.append(marker)
            else:
                source, line_no = location
                evidence.append(Evidence(source, line_no, line_no, f"marker '{marker}' found"))

        if not missing:
            status = CriterionStatus.SATISFIED
            explanation = f"'{requirement}': all markers found ({', '.join(markers)})."
        elif len(missing) == len(markers):
            status = CriterionStatus.NOT_SATISFIED
            explanation = f"'{requirement}': none of the markers found ({', '.join(markers)})."
        else:
            status = CriterionStatus.INSUFFICIENT_EVIDENCE
            explanation = f"'{requirement}': markers not found: {', '.join(missing)}."
        return RubricFeedback(criterion_id, status, explanation, tuple(evidence))


def _find(marker: str, sources: dict[EvidenceSource, list[str]]) -> tuple[EvidenceSource, int] | None:
    """First line containing the marker; code is searched before text."""
    for source, lines in sources.items():
        for line_no, line in enumerate(lines, start=1):
            if marker in line:
                return source, line_no
    return None
