"""One contract test suite for every built-in analyzer (local and model stub alike)."""

from __future__ import annotations

import copy

import pytest

from submission_analyzer.analysis.registry import default_registry
from submission_analyzer.core.config import Mode, ProcessingConfig
from submission_analyzer.core.reports import AnalysisReport, ReportStatus, invalid_evidence
from submission_analyzer.ingestion.validation import validate_record

ANALYZER_IDS = default_registry().ids()

SUBMISSION_VARIANTS = {
    "template_and_solution": {
        "code": (
            "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n\n"
            "    for i in range(1, n + 1):\n        result.append('FizzBuzz' if i % 15 == 0 else 'Fizz' "
            "if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i))\n    return result\n"
        ),
        "text": "I loop from 1 to n.\n\nThen I check divisibility.",
    },
    "no_code": {"code": "", "text": "Only a report, no code."},
    "no_text": {"code": "print('hi')\n", "text": ""},
    "crlf_and_blank_lines": {"code": "\r\n\r\n    return result\r\n", "text": "a\r\n"},
}


@pytest.fixture(params=ANALYZER_IDS)
def analyzer(request):
    return default_registry().create(request.param, ProcessingConfig())


@pytest.fixture(params=sorted(SUBMISSION_VARIANTS))
def submission(request, make_record):
    return validate_record(make_record(**SUBMISSION_VARIANTS[request.param]))


def test_returns_report_with_identification(analyzer, submission, make_context):
    context = make_context()
    report = analyzer.analyze(submission, context)

    assert isinstance(report, AnalysisReport)
    assert report.status is ReportStatus.SUCCEEDED
    assert report.analyzer_id == analyzer.analyzer_id
    assert report.analyzer_version == analyzer.analyzer_version and report.analyzer_version
    assert report.config_hash == context.config_hash
    assert (report.submission_id, report.content_hash) == (submission.submission_id, submission.content_hash)
    assert report.processing_error is None


def test_evidence_points_to_existing_source_lines(analyzer, submission, make_context):
    report = analyzer.analyze(submission, make_context())
    assert invalid_evidence(report, submission) == []


def test_scores_are_in_range_or_null(analyzer, submission, make_context):
    report = analyzer.analyze(submission, make_context())
    for value in (report.score, report.similarity):
        assert value is None or 0.0 <= value <= 1.0


def test_submission_is_not_modified(analyzer, submission, make_context):
    before = copy.deepcopy(submission)
    analyzer.analyze(submission, make_context())
    assert submission == before


def test_result_is_deterministic(analyzer, submission, make_context):
    context = make_context()
    assert analyzer.analyze(submission, context) == analyzer.analyze(submission, context)


def test_report_serializes_to_json_types(analyzer, submission, make_context):
    data = analyzer.analyze(submission, make_context()).to_dict()
    assert data["status"] == "succeeded"
    assert all(isinstance(fb["criterion_id"], str) for fb in data["rubric_feedback"])


@pytest.mark.parametrize("mode", [Mode.ML, Mode.LLM])
def test_works_in_both_modes(mode, submission, make_context):
    config = ProcessingConfig(mode=mode)
    for analyzer_id in ANALYZER_IDS:
        report = default_registry().create(analyzer_id, config).analyze(submission, make_context(config))
        assert report.status is ReportStatus.SUCCEEDED
