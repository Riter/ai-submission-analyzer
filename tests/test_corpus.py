"""The lab 2 training corpus: minimum sizes, provenance and no family leakage between tune and eval."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from submission_analyzer.adapters.loaders import load_assignments, load_pairs, load_provenance, load_split
from submission_analyzer.core.corpus import (
    DerivationRelation,
    LabeledPair,
    Origin,
    Provenance,
    Split,
    SplitName,
    check_no_family_leakage,
)
from submission_analyzer.core.errors import DataSourceError
from submission_analyzer.ingestion.service import IngestionService
from submission_analyzer.storage.repository import InMemorySubmissionRepository

from .conftest import LAB2_DIR, PROJECT_ROOT

FILES = ("assignments.json", "submissions.jsonl", "provenance.jsonl", "pairs.jsonl", "split.json")


@pytest.fixture(scope="module")
def submissions():
    repository = InMemorySubmissionRepository()
    assignments = load_assignments(LAB2_DIR / "assignments.json")
    report = IngestionService(repository, [a.assignment_id for a in assignments]).import_lines(
        (LAB2_DIR / "submissions.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert report.errors == []
    return {s.submission_id: s for s in repository}


@pytest.fixture(scope="module")
def pairs():
    return load_pairs(LAB2_DIR / "pairs.jsonl")


@pytest.fixture(scope="module")
def provenance():
    return load_provenance(LAB2_DIR / "provenance.jsonl")


@pytest.fixture(scope="module")
def split():
    return load_split(LAB2_DIR / "split.json")


def test_generator_is_deterministic_and_matches_committed_corpus(tmp_path: Path):
    script = PROJECT_ROOT / "scripts" / "generate_lab2_data.py"
    for out in ("one", "two"):
        subprocess.run([sys.executable, str(script), "--out", str(tmp_path / out)], check=True, capture_output=True)
    for name in FILES:
        generated = (tmp_path / "one" / name).read_bytes()
        assert generated == (tmp_path / "two" / name).read_bytes(), name
        assert generated == (LAB2_DIR / name).read_bytes(), f"{name} is stale: re-run the generator"


def test_minimum_sizes(submissions, pairs):
    assert len(submissions) >= 30
    assert len(pairs) >= 40
    assert sum(p.template_only for p in pairs) >= 5
    assert len({s.student_id for s in submissions.values()}) >= 30


def test_every_required_kind_of_example_is_present(submissions, pairs, provenance):
    relations = Counter(p.derivation_relation for p in pairs)
    assert all(relations[r] > 0 for r in DerivationRelation)
    origins = Counter(p.origin for p in provenance)
    assert origins[Origin.RESUBMISSION] >= 1
    broken = [s for s in submissions.values() if not _parses(s.code)]
    assert len(broken) == 1


def test_pairs_are_between_different_students_of_one_assignment(submissions, pairs):
    for p in pairs:
        a, b = submissions[p.a], submissions[p.b]
        assert a.assignment_id == b.assignment_id == p.assignment_id
        assert a.student_id != b.student_id


def test_every_submission_has_provenance(submissions, provenance):
    assert {p.submission_id for p in provenance} == set(submissions)
    by_id = {p.submission_id: p for p in provenance}
    for item in provenance:
        if item.derived_from is not None:
            assert by_id[item.derived_from].family_id == item.family_id


def test_no_family_leakage_between_tune_and_eval(split, pairs, provenance):
    assert check_no_family_leakage(split, pairs, provenance) == []
    for side in SplitName:
        assert any(p.split is side for p in pairs)
    tune, evaluation = split.submission_ids(SplitName.TUNE), split.submission_ids(SplitName.EVAL)
    assert tune and evaluation and not (tune & evaluation)


def test_leakage_check_detects_a_family_on_both_sides():
    split = Split(
        families={"f1": SplitName.TUNE, "f2": SplitName.EVAL},
        submissions={"s1": SplitName.TUNE, "s2": SplitName.EVAL},
    )
    leaking = LabeledPair("s1", "s2", "lab", "f1", "f2", DerivationRelation.INDEPENDENT, False, SplitName.TUNE)
    assert check_no_family_leakage(split, [leaking]) != []

    moved = Provenance("s2", "f1", Origin.EXACT_COPY, "s1")  # derivative of f1 placed on the eval side
    assert any("s2" in problem for problem in check_no_family_leakage(split, [], [moved]))


def test_metadata_loaders_reject_bad_files(tmp_path: Path):
    bad = tmp_path / "pairs.jsonl"
    bad.write_text(json.dumps({"a": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(DataSourceError, match="pairs.jsonl:1"):
        load_pairs(bad)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"families": {"f": "test"}, "submissions": {}}), encoding="utf-8")
    with pytest.raises(DataSourceError, match="test"):
        load_split(split)


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True
