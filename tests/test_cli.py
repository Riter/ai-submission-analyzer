"""End-to-end tests of the CLI on the generated lab 1 dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from submission_analyzer.cli import EXIT_OK, EXIT_USAGE, main

from .conftest import DATASET_DIR, PROJECT_ROOT

SUBMISSIONS = str(DATASET_DIR / "submissions.jsonl")
INVALID = str(DATASET_DIR / "invalid_submissions.jsonl")
ASSIGNMENTS = str(DATASET_DIR / "assignments.json")
CONFIG = str(PROJECT_ROOT / "configs" / "default.toml")


def run(capsys, *argv: str) -> tuple[int, dict]:
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out else {}


def test_import_valid_dataset(capsys):
    code, data = run(capsys, "import", SUBMISSIONS, "--assignments", ASSIGNMENTS, "--config", CONFIG)

    assert code == EXIT_OK
    assert set(data) >= {"accepted", "duplicate", "errors", "counts"}
    assert data["counts"] == {"accepted": 12, "duplicate": 1, "rejected_lines": 1}
    assert data["duplicate"][0]["submission_id"] == "sub-0001"
    [conflict] = data["errors"]
    assert (conflict["kind"], conflict["code"], conflict["submission_id"]) == ("conflict", "id_conflict", "sub-0002")


def test_import_invalid_dataset_reports_every_scenario(capsys):
    code, data = run(capsys, "import", INVALID, "--assignments", ASSIGNMENTS)

    assert code == EXIT_OK
    found = {e["code"] for e in data["errors"]}
    assert found == {
        "invalid_json",
        "not_an_object",
        "invalid_timestamp",
        "naive_timestamp",
        "empty_value",
        "unknown_field",
        "invalid_type",
        "missing_field",
        "invalid_ai_declaration",
        "unknown_assignment",
    }
    # The valid last line is still imported after all broken ones.
    assert data["accepted"] == [{"line_no": 12, "submission_id": "ok-0012"}]


def test_import_writes_reproducible_manifest(capsys, tmp_path: Path):
    m1, m2 = tmp_path / "m1.json", tmp_path / "m2.json"
    run(capsys, "import", SUBMISSIONS, "--config", CONFIG, "--manifest", str(m1))
    run(capsys, "import", SUBMISSIONS, "--config", CONFIG, "--manifest", str(m2))

    assert m1.read_bytes() == m2.read_bytes()
    manifest = json.loads(m1.read_text(encoding="utf-8"))
    assert manifest["source"]["file"] == "submissions.jsonl"
    assert manifest["source"]["sha256"].startswith("sha256:")


@pytest.mark.parametrize("mode", ["ml", "llm"])
def test_baseline_command_in_both_modes(capsys, tmp_path: Path, mode):
    config = tmp_path / "c.toml"
    config.write_text(f'[processing]\nmode = "{mode}"\n', encoding="utf-8")

    code, data = run(capsys, "baseline", SUBMISSIONS, "--config", str(config))

    assert code == EXIT_OK
    assert data["mode"] == mode
    assert len(data["results"]) == 12
    assert all(r["mode"] == mode for r in data["results"])


def test_missing_input_file_is_usage_error(capsys, tmp_path: Path):
    code = main(["import", str(tmp_path / "nope.jsonl")])
    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_bad_config_is_usage_error(capsys, tmp_path: Path):
    config = tmp_path / "c.toml"
    config.write_text('[processing]\nmode = "gpt"\n', encoding="utf-8")
    code = main(["import", SUBMISSIONS, "--config", str(config)])
    assert code == EXIT_USAGE
    assert "mode must be one of" in capsys.readouterr().err
