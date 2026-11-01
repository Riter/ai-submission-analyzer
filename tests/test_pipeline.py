"""Lazy pipeline read -> validate -> features -> index."""

from __future__ import annotations

import json
from itertools import islice

from submission_analyzer.adapters.sources import JsonlFileSource
from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.models import Assignment
from submission_analyzer.ingestion.service import ImportReport, IngestionService
from submission_analyzer.similarity.pipeline import build_index, iter_accepted, iter_features
from submission_analyzer.storage.repository import InMemorySubmissionRepository

from .conftest import TEMPLATE

ASSIGNMENTS = {"lab-01-fizzbuzz": Assignment("lab-01-fizzbuzz", "FizzBuzz", (), TEMPLATE)}


class CountingSource:
    """Yields numbered lines and records how many were actually pulled."""

    def __init__(self, lines: list[str], fail_after: int | None = None) -> None:
        self.lines = lines
        self.pulled = 0
        self.fail_after = fail_after

    def __iter__(self):
        for no, line in enumerate(self.lines, start=1):
            if self.fail_after is not None and no > self.fail_after:
                raise AssertionError(f"line {no} was read although nobody asked for it")
            self.pulled += 1
            yield no, line


def lines_for(make_record, n: int) -> list[str]:
    return [json.dumps(make_record(submission_id=f"s{i}", student_id=f"p{i}")) for i in range(n)]


def pipeline(source, report: ImportReport):
    service = IngestionService(InMemorySubmissionRepository())
    return iter_features(iter_accepted(service, source, report), ASSIGNMENTS, SimilarityConfig())


def test_reading_is_lazy(make_record):
    source = CountingSource(lines_for(make_record, 1000), fail_after=2)
    first_two = list(islice(pipeline(source, ImportReport()), 2))
    assert [f.submission_id for f in first_two] == ["s0", "s1"]
    assert source.pulled == 2


def test_nothing_is_read_before_iteration(make_record):
    source = CountingSource(lines_for(make_record, 5))
    stream = pipeline(source, ImportReport())
    assert source.pulled == 0
    next(stream)
    assert source.pulled == 1


def test_bad_lines_are_recorded_and_do_not_stop_the_stream(make_record):
    good = lines_for(make_record, 2)
    report = ImportReport()
    index = build_index(pipeline(CountingSource([good[0], "{broken", "", good[1]]), report))
    assert [f.submission_id for f in index] == ["s0", "s1"]
    assert [(r.line_no, str(r.status)) for r in report.errors] == [(2, "format")]


def test_duplicates_and_conflicts_are_not_indexed_twice(make_record):
    original = make_record(submission_id="s1")
    conflict = make_record(submission_id="s1", code="print('other')\n")
    lines = [json.dumps(original), json.dumps(original), json.dumps(conflict)]
    report = ImportReport()
    index = build_index(pipeline(CountingSource(lines), report))
    assert len(index) == 1
    assert report.to_dict()["counts"] == {"accepted": 1, "duplicate": 1, "rejected_lines": 1}


def test_index_keeps_features_not_raw_content(make_record):
    line = json.dumps(make_record(code=TEMPLATE.replace("return", "result.append('1')\n    return")))
    index = build_index(pipeline(CountingSource([line]), ImportReport()))
    [features] = list(index)
    assert not hasattr(features, "code") and not hasattr(features, "text")
    assert features.template_grams  # template lines were recognized while streaming


def test_file_source_is_closed_after_building_the_index(make_record, tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(lines_for(make_record, 3)) + "\n", encoding="utf-8")
    with JsonlFileSource(path) as source:
        index = build_index(pipeline(source, ImportReport()))
    assert source.closed and len(index) == 3
