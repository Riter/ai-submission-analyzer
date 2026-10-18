"""Deterministic baseline processing.

Both ML and LLM modes are configured, but no model is called yet: each mode
returns the same deterministic result computed from raw content only. The
result records the selected mode and versions so it is traceable later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from submission_analyzer.core.config import Mode, ProcessingConfig
from submission_analyzer.core.models import Submission
from submission_analyzer.core.textutils import count_nonempty_lines


@dataclass(frozen=True, slots=True)
class BaselineResult:
    submission_id: str
    content_hash: str
    mode: Mode
    processing_version: str
    model_version: str | None
    text_nonempty_lines: int
    code_nonempty_lines: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = str(self.mode)
        return data


class BaselineProcessor:
    def __init__(self, config: ProcessingConfig) -> None:
        self._config = config

    def process(self, submission: Submission) -> BaselineResult:
        mode = self._config.mode
        model_version = (
            self._config.prompt_template_version if mode is Mode.LLM else self._config.feature_schema_version
        )
        return BaselineResult(
            submission_id=submission.submission_id,
            content_hash=submission.content_hash,
            mode=mode,
            processing_version=self._config.processing_version,
            model_version=model_version,
            text_nonempty_lines=count_nonempty_lines(submission.text),
            code_nonempty_lines=count_nonempty_lines(submission.code),
            note=f"deterministic baseline for {mode.value} mode; no model was called",
        )
