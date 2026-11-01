from __future__ import annotations

from submission_analyzer.analyzers.base import AnalysisContext, build_report
from submission_analyzer.core.config import ProcessingConfig
from submission_analyzer.core.models import Submission
from submission_analyzer.core.reports import AnalysisReport, RubricFeedback
from submission_analyzer.similarity.features import extract_features
from submission_analyzer.similarity.methods import JaccardMethod
from submission_analyzer.similarity.pairs import PairResult, compare


class TokenSimilarityAnalyzer:
    """Highest token-n-gram Jaccard with another student's submission of the same assignment.

    ``similarity`` is a signal for manual review, not a verdict. Evidence points
    to the lines of *this* submission that share n-grams with the closest one;
    lines of the allowed template are excluded.
    """

    analyzer_id = "token_similarity"
    analyzer_version = "1.0.0"

    def __init__(self, config: ProcessingConfig) -> None:
        self._config = config.similarity
        self._method = JaccardMethod()

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        template = context.assignment.allowed_template
        own = extract_features(submission, template, self._config)
        best: PairResult | None = None
        compared = 0
        for other in context.submissions:
            if other.assignment_id != submission.assignment_id or other.student_id == submission.student_id:
                continue
            compared += 1
            result = compare(own, extract_features(other, template, self._config), self._method)
            if result.score is not None and (best is None or (result.score, best.b) > (best.score, result.b)):
                best = result

        if best is None:
            reason = (
                f"code could not be tokenized ({own.parse_error})"
                if own.parse_error
                else "no other student's submission of this assignment to compare with"
                if compared == 0
                else "not enough n-grams outside the allowed template"
            )
            feedback = RubricFeedback("similarity", None, f"Similarity not computed: {reason}.")
            return build_report(self, submission, context, similarity=None, rubric_feedback=(feedback,))

        feedback = RubricFeedback(
            "similarity",
            None,
            f"Closest submission: {best.b} (token {self._config.ngram}-gram Jaccard {best.score}). {best.explanation}",
        )
        return build_report(
            self, submission, context, similarity=best.score, rubric_feedback=(feedback,), evidence=best.evidence_a
        )
