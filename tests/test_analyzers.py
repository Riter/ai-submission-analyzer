"""Behaviour of each concrete analyzer."""

from __future__ import annotations

import json

import pytest

from submission_analyzer.analyzers import LengthAnalyzer, ModelStubAnalyzer, RequirementsAnalyzer, TemplateAnalyzer
from submission_analyzer.analyzers.model_stub import JsonFeedbackFormatter, ModelOutputError, RubricInputPreparer
from submission_analyzer.analyzers.requirements import requirement_markers
from submission_analyzer.analyzers.template import TEMPLATE_NOTE
from submission_analyzer.core.config import Mode, ProcessingConfig
from submission_analyzer.core.reports import CriterionStatus, EvidenceSource
from submission_analyzer.ingestion.validation import validate_record

from .conftest import TEMPLATE

# ---------- LengthAnalyzer ----------


def test_length_counts_nonempty_lines_without_score(make_record, make_context):
    submission = validate_record(make_record(text="a\n\n  \nb\nc", code="x = 1\n\n\ny = 2\n"))
    report = LengthAnalyzer().analyze(submission, make_context())

    assert report.score is None
    [feedback] = report.rubric_feedback
    assert feedback.status is None
    assert "3 non-empty lines" in feedback.explanation
    assert "code has 2 non-empty lines" in feedback.explanation


# ---------- TemplateAnalyzer ----------


def test_template_only_code_has_full_similarity(make_record, make_context):
    report = TemplateAnalyzer().analyze(validate_record(make_record(code=TEMPLATE)), make_context())

    assert report.similarity == 1.0
    assert [(e.start_line, e.end_line) for e in report.evidence] == [(1, 3)]
    assert report.evidence[0].note == TEMPLATE_NOTE
    assert "permitted" in report.rubric_feedback[0].explanation


def test_template_line_numbers_refer_to_original_content(make_record, make_context):
    code = (
        "# my solution\n"  # 1
        "\n"  # 2
        "def fizzbuzz(n: int) -> list[str]:\n"  # 3  template
        "    result: list[str] = []\n"  # 4  template
        "    for i in range(n):\n"  # 5
        "        result.append(str(i))\n"  # 6
        "\n"  # 7
        "    return result\n"  # 8  template
    )
    report = TemplateAnalyzer().analyze(validate_record(make_record(code=code)), make_context())

    assert [(e.start_line, e.end_line) for e in report.evidence] == [(3, 4), (8, 8)]
    assert all(e.source is EvidenceSource.CODE for e in report.evidence)
    assert report.similarity == round(3 / 6, 4)


def test_template_match_ignores_indentation_changes(make_record, make_context):
    code = "def fizzbuzz(n: int) -> list[str]:\n        return result\n"
    report = TemplateAnalyzer().analyze(validate_record(make_record(code=code)), make_context())
    assert report.similarity == 1.0


def test_no_template_lines_gives_zero_similarity(make_record, make_context):
    report = TemplateAnalyzer().analyze(validate_record(make_record(code="print(42)\n")), make_context())
    assert report.similarity == 0.0
    assert report.evidence == ()


def test_template_skipped_without_code(make_record, make_context):
    report = TemplateAnalyzer().analyze(validate_record(make_record(code="")), make_context())
    assert report.similarity is None
    assert "no code" in report.rubric_feedback[0].explanation


# ---------- RequirementsAnalyzer ----------


def test_requirement_markers():
    assert requirement_markers("Function fizzbuzz(n) returns a list") == ("fizzbuzz",)
    assert requirement_markers("Multiples are 'Fizz' and 'Buzz'") == ("Fizz", "Buzz")
    assert requirement_markers("Report explains the approach") == ()


def test_requirements_statuses_and_evidence(make_record, make_context):
    code = "def fizzbuzz(n):\n    return ['Fizz', 'Buzz']\n"
    report = RequirementsAnalyzer().analyze(validate_record(make_record(code=code)), make_context())
    by_id = {fb.criterion_id: fb for fb in report.rubric_feedback}

    assert by_id["req-1"].status is CriterionStatus.SATISFIED
    assert [(e.source, e.start_line) for e in by_id["req-1"].evidence] == [(EvidenceSource.CODE, 1)]
    assert by_id["req-2"].status is CriterionStatus.INSUFFICIENT_EVIDENCE  # 'FizzBuzz' is missing
    assert by_id["req-3"].status is CriterionStatus.INSUFFICIENT_EVIDENCE  # nothing checkable


def test_requirements_not_satisfied_when_nothing_found(make_record, make_context):
    report = RequirementsAnalyzer().analyze(validate_record(make_record(code="print(1)\n")), make_context())
    assert report.rubric_feedback[0].status is CriterionStatus.NOT_SATISFIED


# ---------- ModelStubAnalyzer ----------


@pytest.mark.parametrize(
    ("config", "version"),
    [
        (ProcessingConfig(mode=Mode.LLM, prompt_template_version="prompt-3"), "0.1.0+llm.prompt-3"),
        (ProcessingConfig(mode=Mode.ML, feature_schema_version="feat-2"), "0.1.0+ml.feat-2"),
    ],
)
def test_model_stub_version_tracks_prompt_or_feature_schema(config, version):
    assert ModelStubAnalyzer(config).analyzer_version == version


def test_model_stub_covers_every_criterion(make_record, make_context, assignment):
    report = ModelStubAnalyzer(ProcessingConfig()).analyze(validate_record(make_record()), make_context())
    assert [fb.criterion_id for fb in report.rubric_feedback] == ["req-1", "req-2", "req-3"]
    assert all(fb.status is CriterionStatus.INSUFFICIENT_EVIDENCE for fb in report.rubric_feedback)


class _FixedClient:
    def __init__(self, raw: str) -> None:
        self.raw = raw

    def complete(self, model_input) -> str:
        return self.raw


def _feedback(criteria, **extra):
    return json.dumps({"criteria": [{"criterion_id": c, "status": "satisfied", "explanation": "ok", **extra} for c in criteria]})


def test_model_client_is_replaceable_without_changing_the_contract(make_record, make_context):
    evidence = [{"source": "code", "start_line": 1, "end_line": 2, "note": "solution"}]
    client = _FixedClient(_feedback(["req-1", "req-2", "req-3"], evidence=evidence))
    report = ModelStubAnalyzer(ProcessingConfig(), client=client).analyze(validate_record(make_record()), make_context())

    assert all(fb.status is CriterionStatus.SATISFIED for fb in report.rubric_feedback)
    assert report.rubric_feedback[0].evidence[0].end_line == 2


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"answer": 1}', "'criteria' array"),
        (_feedback(["req-1", "req-2"]), "missing criteria"),
        (_feedback(["req-1", "req-2", "req-3", "req-9"]), "unknown criterion"),
        (_feedback(["req-1", "req-1", "req-2", "req-3"]), "returned twice"),
        (json.dumps({"criteria": [{"criterion_id": "req-1", "status": "great", "explanation": ""}]}), "invalid status"),
        (_feedback(["req-1", "req-2", "req-3"], evidence=[{"source": "code", "start_line": 1, "end_line": 99}]), "outside"),
        (_feedback(["req-1", "req-2", "req-3"], evidence=[{"source": "image", "start_line": 1, "end_line": 1}]), "malformed"),
    ],
)
def test_invalid_model_output_is_rejected(make_record, make_context, raw, message):
    submission = validate_record(make_record())
    model_input = RubricInputPreparer().prepare(submission, make_context())
    with pytest.raises(ModelOutputError, match=message):
        JsonFeedbackFormatter().format(raw, model_input)
