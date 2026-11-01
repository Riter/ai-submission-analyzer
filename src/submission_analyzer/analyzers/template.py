from __future__ import annotations

from submission_analyzer.analyzers.base import AnalysisContext, build_report
from submission_analyzer.core.models import Submission
from submission_analyzer.core.reports import AnalysisReport, Evidence, EvidenceSource, RubricFeedback
from submission_analyzer.core.textutils import line_runs, source_lines

TEMPLATE_NOTE = "exact match with the allowed assignment template (permitted, not a violation)"


class TemplateAnalyzer:
    """Finds code lines that exactly match lines of the allowed template.

    ``similarity`` is the share of non-empty code lines that match the template
    (comparison ignores leading/trailing whitespace). Matching the template is
    explicitly allowed, so this is context for a reviewer, not a finding.
    """

    analyzer_id = "template"
    analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        template = {line.strip() for line in source_lines(context.assignment.allowed_template) if line.strip()}
        code = source_lines(submission.code)
        nonempty = [no for no, line in enumerate(code, start=1) if line.strip()]

        if not template or not nonempty:
            reason = "assignment has no allowed template" if not template else "submission has no code"
            feedback = RubricFeedback("allowed_template", None, f"Template comparison skipped: {reason}.")
            return build_report(self, submission, context, similarity=None, rubric_feedback=(feedback,))

        matched = [no for no in nonempty if code[no - 1].strip() in template]
        evidence = tuple(
            Evidence(EvidenceSource.CODE, start, end, TEMPLATE_NOTE) for start, end in line_runs(matched)
        )
        similarity = round(len(matched) / len(nonempty), 4)
        feedback = RubricFeedback(
            criterion_id="allowed_template",
            status=None,
            explanation=(
                f"{len(matched)} of {len(nonempty)} non-empty code lines exactly match the allowed template. "
                "Using the allowed template is permitted and is not evidence of copying."
            ),
        )
        return build_report(
            self, submission, context, similarity=similarity, rubric_feedback=(feedback,), evidence=evidence
        )

