"""One contract suite for both submission sources, plus resource-release checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from submission_analyzer.adapters.sources import InMemorySource, JsonlFileSource
from submission_analyzer.ingestion.service import IngestionService
from submission_analyzer.storage.repository import InMemorySubmissionRepository

LINES = ['{"a": 1}', "", "not json", '{"b": 2}']


@pytest.fixture(params=["jsonl", "memory"])
def make_source(request, tmp_path: Path):
    def factory(lines: list[str]):
        if request.param == "memory":
            return InMemorySource(lines)
        path = tmp_path / "src.jsonl"
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return JsonlFileSource(path)

    return factory


def test_yields_numbered_lines(make_source):
    with make_source(LINES) as source:
        assert list(source) == [(1, '{"a": 1}'), (2, ""), (3, "not json"), (4, '{"b": 2}')]


def test_closed_after_with_block(make_source):
    with make_source(LINES) as source:
        assert not source.closed
    assert source.closed


def test_iteration_outside_with_is_an_error(make_source):
    source = make_source(LINES)
    with pytest.raises(RuntimeError, match="with"):
        list(source)


def test_closed_on_exception(make_source):
    source = make_source(LINES)
    with pytest.raises(ZeroDivisionError):
        with source:
            for _ in source:
                1 / 0
    assert source.closed


def test_closed_on_early_exit(make_source):
    with make_source(LINES) as source:
        for line_no, _ in source:
            if line_no == 1:
                break
    assert source.closed


def test_same_import_result_from_both_sources(make_source, make_record):
    lines = [json.dumps(make_record(submission_id="a")), "{broken", json.dumps(make_record(submission_id="b"))]
    service = IngestionService(InMemorySubmissionRepository())
    with make_source(lines) as source:
        report = service.import_numbered(source)
    data = report.to_dict()
    assert data["counts"] == {"accepted": 2, "duplicate": 0, "rejected_lines": 1}
    assert data["errors"][0]["line_no"] == 2


def test_file_closed_when_import_raises(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"a": 1}\n\xff\xfe broken utf-8\n')
    source = JsonlFileSource(path)
    service = IngestionService(InMemorySubmissionRepository())
    with pytest.raises(UnicodeDecodeError):
        with source:
            service.import_numbered(source)
    assert source.closed


def test_real_file_handle_is_closed(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    source = JsonlFileSource(path)
    with source:
        handle = source._fh
    assert handle is not None and handle.closed
