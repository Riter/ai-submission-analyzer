"""End-to-end tests of the CLI on the generated lab 1 dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from submission_analyzer.cli import EXIT_OK, EXIT_USAGE, main

from .conftest import DATASET_DIR, LAB2_DIR, PROJECT_ROOT

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


def test_analyze_command(capsys):
    code, data = run(capsys, "analyze", SUBMISSIONS, "--assignments", ASSIGNMENTS, "--config", CONFIG)

    assert code == EXIT_OK
    assert data["import"] == {"accepted": 12, "duplicate": 1, "rejected_lines": 1}
    assert len(data["reports"]) == 12 * 5
    assert {r["analyzer_id"] for r in data["reports"]} == {
        "length",
        "template",
        "requirements",
        "token_similarity",
        "model_stub",
    }
    assert all(r["status"] == "succeeded" and r["config_hash"] == data["config_hash"] for r in data["reports"])


def test_analyze_is_reproducible_and_mode_changes_config_hash(capsys, tmp_path: Path):
    _, first = run(capsys, "analyze", SUBMISSIONS, "--assignments", ASSIGNMENTS, "--config", CONFIG)
    _, second = run(capsys, "analyze", SUBMISSIONS, "--assignments", ASSIGNMENTS, "--config", CONFIG)
    assert first == second

    llm = tmp_path / "llm.toml"
    llm.write_text('[processing]\nmode = "llm"\n', encoding="utf-8")
    _, llm_run = run(capsys, "analyze", SUBMISSIONS, "--assignments", ASSIGNMENTS, "--config", str(llm))
    assert llm_run["config_hash"] != first["config_hash"]
    stub_versions = {r["analyzer_version"] for r in llm_run["reports"] if r["analyzer_id"] == "model_stub"}
    assert stub_versions == {"0.1.0+llm.stub-0"}


def test_analyze_requires_assignments(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["analyze", SUBMISSIONS])
    assert exc_info.value.code == EXIT_USAGE
    assert "requires --assignments" in capsys.readouterr().err


def test_analyze_unknown_analyzer_is_usage_error(capsys, tmp_path: Path):
    config = tmp_path / "c.toml"
    config.write_text('[processing]\n[analyzers]\nenabled = ["length", "nope"]\n', encoding="utf-8")
    code = main(["analyze", SUBMISSIONS, "--assignments", ASSIGNMENTS, "--config", str(config)])
    assert code == EXIT_USAGE
    assert "unknown analyzer 'nope'" in capsys.readouterr().err


# --- lab 2: similarity on the generated corpus ---

LAB2 = str(LAB2_DIR / "submissions.jsonl")
LAB2_ASSIGNMENTS = str(LAB2_DIR / "assignments.json")
LAB2_SPLIT = str(LAB2_DIR / "split.json")


def labeled_pairs() -> dict[tuple[str, str], dict]:
    lines = (LAB2_DIR / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    return {(p["a"], p["b"]): p for p in map(json.loads, lines)}


@pytest.mark.parametrize("method", ["jaccard", "tfidf"])
def test_similarity_command_on_corpus(capsys, method):
    code, data = run(
        capsys, "similarity", LAB2, "--assignments", LAB2_ASSIGNMENTS, "--config", CONFIG,
        "--method", method, "--split", LAB2_SPLIT,
    )  # fmt: skip
    assert code == EXIT_OK
    assert data["method"]["id"] == method and data["method"]["fit"]["side"] == "tune"
    assert data["import"]["accepted"] == 40

    labels = labeled_pairs()
    scored = {(r["a"], r["b"]): r for r in data["pairs"]}
    copies = [scored[k]["score"] for k, p in labels.items() if p["derivation_relation"] == "exact_copy"]
    assert copies and all(score == 1.0 for score in copies)
    assert all(r["a"] != r["b"] for r in data["pairs"])
    assert {r["kind"] for r in data["same_student_pairs"]} == {"same_student"}


def test_similarity_top_k(capsys):
    code, data = run(
        capsys, "similarity", LAB2, "--assignments", LAB2_ASSIGNMENTS, "--submission", "sub-0001", "--top-k", "3",
    )  # fmt: skip
    assert code == EXIT_OK
    assert len(data["top_k"]) == 3
    assert all(r["a"] == "sub-0001" and r["b"] != "sub-0001" for r in data["top_k"])
    assert len({r["assignment_id"] for r in data["top_k"]}) == 1


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--method", "tfidf"], "requires --split"),
        (["--submission", "nope"], "not among accepted"),
        (["--top-k", "0"], "positive"),
    ],
)
def test_similarity_usage_errors(capsys, extra, message):
    code = main(["similarity", LAB2, "--assignments", LAB2_ASSIGNMENTS, *extra])
    assert code == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_similarity_method_must_be_enabled(capsys, tmp_path: Path):
    config = tmp_path / "jaccard_only.toml"
    config.write_text('[processing]\n[similarity]\nmethods = ["jaccard"]\n', encoding="utf-8")
    code = main(["similarity", LAB2, "--assignments", LAB2_ASSIGNMENTS, "--config", str(config),
                 "--method", "tfidf", "--split", LAB2_SPLIT])  # fmt: skip
    assert code == EXIT_USAGE
    assert "not enabled" in capsys.readouterr().err
