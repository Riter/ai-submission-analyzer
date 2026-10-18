"""Model-backed analyzer split into replaceable parts.

    submission ──InputPreparer──> ModelInput ──ModelClient──> raw JSON ──FeedbackFormatter──> RubricFeedback

In lab 1 the client is a deterministic stub: no network, no real model. A real
LLM client (lab 2+) only has to implement :class:`ModelClient`; preparation,
output validation and the report contract stay the same.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from submission_analyzer.analyzers.base import AnalysisContext, build_report
from submission_analyzer.core.config import Mode, ProcessingConfig
from submission_analyzer.core.models import Submission
from submission_analyzer.core.reports import (
    AnalysisReport,
    CriterionStatus,
    Evidence,
    EvidenceSource,
    RubricFeedback,
)
from submission_analyzer.core.textutils import source_lines

ANALYZER_BASE_VERSION = "0.1.0"


class ModelOutputError(ValueError):
    """The model returned something that does not match the expected structure."""


@dataclass(frozen=True, slots=True)
class Criterion:
    criterion_id: str
    description: str


@dataclass(frozen=True, slots=True)
class ModelInput:
    mode: Mode
    # prompt template version (llm) or feature schema version (ml)
    model_version: str
    criteria: tuple[Criterion, ...]
    text_lines: tuple[str, ...]
    code_lines: tuple[str, ...]


class InputPreparer(Protocol):
    def prepare(self, submission: Submission, context: AnalysisContext) -> ModelInput: ...


class ModelClient(Protocol):
    def complete(self, model_input: ModelInput) -> str: ...


class FeedbackFormatter(Protocol):
    def format(self, raw_output: str, model_input: ModelInput) -> tuple[RubricFeedback, ...]: ...


def model_version_for(config: ProcessingConfig) -> str:
    version = config.prompt_template_version if config.mode is Mode.LLM else config.feature_schema_version
    if not version:
        raise ValueError(f"no model version configured for {config.mode} mode")
    return version


class RubricInputPreparer:
    """Turns assignment requirements into criteria and passes content as numbered lines."""

    def prepare(self, submission: Submission, context: AnalysisContext) -> ModelInput:
        return ModelInput(
            mode=context.config.mode,
            model_version=model_version_for(context.config),
            criteria=tuple(
                Criterion(f"req-{i}", text) for i, text in enumerate(context.assignment.requirements, start=1)
            ),
            text_lines=tuple(source_lines(submission.text)),
            code_lines=tuple(source_lines(submission.code)),
        )


class StubModelClient:
    """Deterministic stand-in for a model: marks every criterion as needing review."""

    def complete(self, model_input: ModelInput) -> str:
        return json.dumps(
            {
                "criteria": [
                    {
                        "criterion_id": c.criterion_id,
                        "status": CriterionStatus.INSUFFICIENT_EVIDENCE.value,
                        "explanation": "stub model: no real model was called; manual review required",
                        "evidence": [],
                    }
                    for c in model_input.criteria
                ]
            }
        )


class JsonFeedbackFormatter:
    """Validates the model's JSON output before it becomes feedback.

    Model output is untrusted: unknown criteria, invalid statuses and evidence
    pointing at non-existent lines are rejected with :class:`ModelOutputError`.
    """

    def format(self, raw_output: str, model_input: ModelInput) -> tuple[RubricFeedback, ...]:
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ModelOutputError(f"model output is not valid JSON: {exc.msg}") from exc
        items = data.get("criteria") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ModelOutputError("model output must be an object with a 'criteria' array")

        expected = {c.criterion_id for c in model_input.criteria}
        seen: set[str] = set()
        feedback: list[RubricFeedback] = []
        for item in items:
            fb = self._parse_item(item, model_input)
            if fb.criterion_id not in expected:
                raise ModelOutputError(f"unknown criterion '{fb.criterion_id}'")
            if fb.criterion_id in seen:
                raise ModelOutputError(f"criterion '{fb.criterion_id}' returned twice")
            seen.add(fb.criterion_id)
            feedback.append(fb)
        if seen != expected:
            raise ModelOutputError(f"missing criteria: {', '.join(sorted(expected - seen))}")
        return tuple(feedback)

    def _parse_item(self, item: object, model_input: ModelInput) -> RubricFeedback:
        if not isinstance(item, dict):
            raise ModelOutputError("each criterion must be an object")
        criterion_id, status, explanation = item.get("criterion_id"), item.get("status"), item.get("explanation")
        if not isinstance(criterion_id, str) or not isinstance(explanation, str):
            raise ModelOutputError("criterion_id and explanation must be strings")
        try:
            parsed_status = CriterionStatus(status)
        except ValueError:
            raise ModelOutputError(f"invalid status {status!r} for '{criterion_id}'") from None

        limits = {EvidenceSource.TEXT: len(model_input.text_lines), EvidenceSource.CODE: len(model_input.code_lines)}
        evidence: list[Evidence] = []
        for ev in item.get("evidence", []):
            try:
                source = EvidenceSource(ev["source"])
                start, end = int(ev["start_line"]), int(ev["end_line"])
            except (KeyError, TypeError, ValueError):
                raise ModelOutputError(f"malformed evidence for '{criterion_id}'") from None
            if not 1 <= start <= end <= limits[source]:
                raise ModelOutputError(f"evidence {source} {start}-{end} for '{criterion_id}' points outside the submission")
            evidence.append(Evidence(source, start, end, str(ev.get("note", ""))))
        return RubricFeedback(criterion_id, parsed_status, explanation, tuple(evidence))


class ModelStubAnalyzer:
    """Model-backed analyzer with the same report contract as the local ones."""

    analyzer_id = "model_stub"

    def __init__(
        self,
        config: ProcessingConfig,
        preparer: InputPreparer | None = None,
        client: ModelClient | None = None,
        formatter: FeedbackFormatter | None = None,
    ) -> None:
        self._preparer = preparer or RubricInputPreparer()
        self._client = client or StubModelClient()
        self._formatter = formatter or JsonFeedbackFormatter()
        # e.g. "0.1.0+llm.stub-0": analyzer code version + mode + prompt/feature-schema version
        self.analyzer_version = f"{ANALYZER_BASE_VERSION}+{config.mode}.{model_version_for(config)}"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        model_input = self._preparer.prepare(submission, context)
        raw_output = self._client.complete(model_input)
        feedback = self._formatter.format(raw_output, model_input)
        return build_report(self, submission, context, rubric_feedback=feedback)
