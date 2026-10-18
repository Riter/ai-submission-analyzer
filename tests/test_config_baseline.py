from __future__ import annotations

from pathlib import Path

import pytest

from submission_analyzer.baseline import BaselineProcessor, count_nonempty_lines
from submission_analyzer.config import Mode, ProcessingConfig, load_config, parse_config
from submission_analyzer.errors import ConfigError
from submission_analyzer.validation import validate_record

from .conftest import PROJECT_ROOT


def test_default_config_file_loads():
    config = load_config(PROJECT_ROOT / "configs" / "default.toml")
    assert config.mode is Mode.LLM
    assert config.processing_version == "1.0.0"


@pytest.mark.parametrize("mode", ["ml", "llm"])
def test_both_modes_parse(mode):
    assert parse_config({"processing": {"mode": mode}}).mode == Mode(mode)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "[processing]"),
        ({"processing": {"mode": "gpt"}}, "mode must be one of"),
        ({"processing": {"mode": "llm", "temperature": 0.2}}, "unknown config keys"),
        ({"processing": {"processing_version": ""}}, "processing_version"),
        ({"processing": {"mode": "llm", "prompt_template_version": None}}, "prompt_template_version"),
    ],
)
def test_invalid_config_is_rejected(data, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[")):
        parse_config(data)


def test_unreadable_and_malformed_config_files(tmp_path: Path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[processing\nmode = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)


def test_count_nonempty_lines():
    assert count_nonempty_lines("a\n\n  \nb\n") == 2
    assert count_nonempty_lines("") == 0


@pytest.mark.parametrize("mode", [Mode.ML, Mode.LLM])
def test_baseline_is_deterministic_in_both_modes(make_record, mode):
    submission = validate_record(make_record())
    processor = BaselineProcessor(ProcessingConfig(mode=mode))

    first, second = processor.process(submission), processor.process(submission)

    assert first == second
    assert first.mode is mode
    assert (first.text_nonempty_lines, first.code_nonempty_lines) == (2, 2)
    assert "no model was called" in first.note


def test_baseline_records_version_of_the_selected_mode(make_record):
    submission = validate_record(make_record())
    config = ProcessingConfig(prompt_template_version="prompt-7", feature_schema_version="features-3")

    llm = BaselineProcessor(config).process(submission)
    ml = BaselineProcessor(ProcessingConfig(mode=Mode.ML, feature_schema_version="features-3")).process(submission)

    assert llm.model_version == "prompt-7"
    assert ml.model_version == "features-3"
    # Content-derived part of the result does not depend on the mode.
    assert (llm.text_nonempty_lines, llm.code_nonempty_lines) == (ml.text_nonempty_lines, ml.code_nonempty_lines)
