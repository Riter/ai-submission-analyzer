from __future__ import annotations

from submission_analyzer.analyzers.base import AnalysisContext, build_report
from submission_analyzer.core.models import Submission
from submission_analyzer.core.reports import AnalysisReport, RubricFeedback
from submission_analyzer.core.textutils import count_nonempty_lines


class LengthAnalyzer:
    """Counts non-empty lines of text and code. Informational only: no score."""

    analyzer_id = "length"
    analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        text_lines = count_nonempty_lines(submission.text)
        code_lines = count_nonempty_lines(submission.code)
        feedback = RubricFeedback(
            criterion_id="length",
            status=None,
            explanation=(
                f"Report has {text_lines} non-empty lines, code has {code_lines} non-empty lines. "
                "Length is informational and is not scored."
            ),
        )
        return build_report(self, submission, context, score=None, rubric_feedback=(feedback,))
