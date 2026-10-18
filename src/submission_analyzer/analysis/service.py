"""Central analysis service.

Runs every configured analyzer on every submission through the common
:class:`Analyzer` contract. It knows analyzers only by id (via the registry)
and never branches on a concrete analyzer class.
"""

from __future__ import annotations

from collections.abc import Mapping

from submission_analyzer.analysis.registry import AnalyzerRegistry
from submission_analyzer.analyzers import AnalysisContext, Analyzer, SubmissionReader
from submission_analyzer.core.config import ProcessingConfig, config_hash
from submission_analyzer.core.models import Assignment, Submission
from submission_analyzer.core.reports import AnalysisReport, ReportStatus, invalid_evidence


class AnalysisService:
    def __init__(
        self,
        registry: AnalyzerRegistry,
        config: ProcessingConfig,
        assignments: Mapping[str, Assignment],
        submissions: SubmissionReader,
    ) -> None:
        # Unknown analyzer ids fail here, before any submission is processed.
        self._analyzers: tuple[Analyzer, ...] = tuple(registry.create(a, config) for a in config.analyzers)
        self._config = config
        self._config_hash = config_hash(config)
        self._assignments = assignments
        self._submissions = submissions

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def analyze(self, submission: Submission) -> list[AnalysisReport]:
        """Run all analyzers on one submission.

        A failing analyzer yields a ``failed`` report with ``processing_error``;
        the other analyzers still run.
        """
        assignment = self._assignments.get(submission.assignment_id)
        if assignment is None:
            return [
                self._failed(a, submission, f"assignment '{submission.assignment_id}' is not loaded")
                for a in self._analyzers
            ]
        context = AnalysisContext(assignment, self._submissions, self._config, self._config_hash)
        return [self._run(analyzer, submission, context) for analyzer in self._analyzers]

    def analyze_all(self) -> list[AnalysisReport]:
        return [report for submission in self._submissions for report in self.analyze(submission)]

    def _run(self, analyzer: Analyzer, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        try:
            report = analyzer.analyze(submission, context)
        except Exception as exc:  # noqa: BLE001 — isolate any analyzer failure
            return self._failed(analyzer, submission, f"{type(exc).__name__}: {exc}")

        if not isinstance(report, AnalysisReport):
            return self._failed(analyzer, submission, f"analyzer returned {type(report).__name__}, not AnalysisReport")
        if (report.analyzer_id, report.submission_id) != (analyzer.analyzer_id, submission.submission_id):
            return self._failed(analyzer, submission, "report does not belong to this analyzer/submission")
        problems = invalid_evidence(report, submission)
        if problems:
            return self._failed(analyzer, submission, "invalid evidence: " + "; ".join(problems))
        return report

    def _failed(self, analyzer: Analyzer, submission: Submission, error: str) -> AnalysisReport:
        return AnalysisReport(
            submission_id=submission.submission_id,
            content_hash=submission.content_hash,
            analyzer_id=analyzer.analyzer_id,
            analyzer_version=analyzer.analyzer_version,
            config_hash=self._config_hash,
            status=ReportStatus.FAILED,
            processing_error=error,
        )
