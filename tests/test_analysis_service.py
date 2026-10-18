"""Central analysis service: registry lookup, failure isolation, extensibility."""

from __future__ import annotations

import inspect

import pytest

import submission_analyzer.analysis.service as analysis_module
from submission_analyzer.analysis.registry import AnalyzerRegistry, default_registry
from submission_analyzer.analysis.service import AnalysisService
from submission_analyzer.analyzers import AnalysisContext, build_report
from submission_analyzer.core.config import ProcessingConfig, config_hash
from submission_analyzer.core.errors import ConfigError
from submission_analyzer.core.models import Submission
from submission_analyzer.core.reports import (
    AnalysisReport,
    CriterionStatus,
    Evidence,
    EvidenceSource,
    ReportStatus,
    RubricFeedback,
)
from submission_analyzer.ingestion.validation import validate_record


@pytest.fixture
def stored(repository, make_record):
    submission = validate_record(make_record())
    repository.add(submission)
    return submission


def make_service(repository, assignment, registry=None, analyzers=("length", "template")):
    config = ProcessingConfig(analyzers=tuple(analyzers))
    return AnalysisService(registry or default_registry(), config, {assignment.assignment_id: assignment}, repository)


def test_runs_configured_analyzers_in_order(repository, assignment, stored):
    reports = make_service(repository, assignment).analyze_all()
    assert [r.analyzer_id for r in reports] == ["length", "template"]
    assert {r.config_hash for r in reports} == {config_hash(ProcessingConfig(analyzers=("length", "template")))}


def test_unknown_analyzer_fails_before_processing(repository, assignment):
    with pytest.raises(ConfigError, match="unknown analyzer 'nope'"):
        make_service(repository, assignment, analyzers=("length", "nope"))


def test_registry_rejects_duplicate_and_mismatched_ids():
    registry = AnalyzerRegistry()
    registry.register("length", lambda config: default_registry().create("length", config))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("length", lambda config: None)
    registry.register("alias", lambda config: default_registry().create("length", config))
    with pytest.raises(ConfigError, match="produced analyzer 'length'"):
        registry.create("alias", ProcessingConfig())


class _CrashingAnalyzer:
    analyzer_id = "crashing"
    analyzer_version = "0.0.1"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        raise RuntimeError("boom")


class _BadEvidenceAnalyzer:
    analyzer_id = "bad_evidence"
    analyzer_version = "0.0.1"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        return build_report(self, submission, context, evidence=(Evidence(EvidenceSource.CODE, 1, 500, "x"),))


def _registry_with(*extra) -> AnalyzerRegistry:
    registry = default_registry()
    for analyzer_cls in extra:
        registry.register(analyzer_cls.analyzer_id, lambda config, cls=analyzer_cls: cls())
    return registry


def test_analyzer_error_does_not_stop_other_analyzers(repository, assignment, stored):
    service = make_service(
        repository, assignment, _registry_with(_CrashingAnalyzer), analyzers=("crashing", "length")
    )
    crashed, length = service.analyze_all()

    assert crashed.status is ReportStatus.FAILED
    assert crashed.processing_error == "RuntimeError: boom"
    assert crashed.analyzer_version == "0.0.1"
    assert length.status is ReportStatus.SUCCEEDED


def test_report_with_out_of_range_evidence_is_rejected(repository, assignment, stored):
    service = make_service(repository, assignment, _registry_with(_BadEvidenceAnalyzer), analyzers=("bad_evidence",))
    [report] = service.analyze_all()
    assert report.status is ReportStatus.FAILED
    assert "invalid evidence" in report.processing_error


def test_missing_assignment_gives_failed_reports(repository, assignment, make_record):
    submission = validate_record(make_record(assignment_id="lab-02-word-count"))
    repository.add(submission)
    reports = make_service(repository, assignment).analyze(submission)
    assert all(r.status is ReportStatus.FAILED and "not loaded" in r.processing_error for r in reports)


def test_service_has_no_branching_on_concrete_analyzers():
    source = inspect.getsource(analysis_module)
    for name in ("LengthAnalyzer", "TemplateAnalyzer", "RequirementsAnalyzer", "ModelStubAnalyzer"):
        assert name not in source


# --- Defense rehearsal: a third analyzer plugged in without touching AnalysisService ---


class AIDeclarationAnalyzer:
    """Checks that a used=true AI declaration has a non-empty description."""

    analyzer_id = "ai_declaration"
    analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        declaration = submission.ai_declaration
        if not declaration.used:
            status, explanation = CriterionStatus.SATISFIED, "AI use not declared; nothing to describe."
        elif declaration.description.strip():
            status, explanation = CriterionStatus.SATISFIED, "AI use is declared and described."
        else:
            status, explanation = CriterionStatus.NOT_SATISFIED, "AI use is declared but the description is empty."
        return build_report(
            self, submission, context, rubric_feedback=(RubricFeedback("ai_declaration", status, explanation),)
        )


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"used": True, "tools": ["Claude"], "description": "Asked for tests."}, CriterionStatus.SATISFIED),
        ({"used": True, "tools": [], "description": "  "}, CriterionStatus.NOT_SATISFIED),
        ({"used": False, "tools": [], "description": ""}, CriterionStatus.SATISFIED),
    ],
)
def test_third_analyzer_plugs_in_via_registry(repository, assignment, make_record, declaration, expected):
    submission = validate_record(make_record(ai_declaration=declaration))
    repository.add(submission)
    service = make_service(
        repository, assignment, _registry_with(AIDeclarationAnalyzer), analyzers=("length", "ai_declaration")
    )
    _, report = service.analyze(submission)

    assert report.analyzer_id == "ai_declaration"
    assert report.rubric_feedback[0].status is expected
