from __future__ import annotations

import json

from submission_analyzer.core.config import ProcessingConfig
from submission_analyzer.ingestion.manifest import build_manifest
from submission_analyzer.ingestion.service import IngestionService
from submission_analyzer.storage.repository import InMemorySubmissionRepository


def _import(lines):
    return IngestionService(InMemorySubmissionRepository()).import_lines(lines)


def test_manifest_lists_ids_hashes_and_versions(make_record):
    lines = [json.dumps(make_record()), json.dumps(make_record()), "{broken"]
    report = _import(lines)

    manifest = build_manifest(report, ProcessingConfig(), source_name="s.jsonl", source_sha256="sha256:abc")

    assert manifest["processing"]["processing_version"] == "1.0.0"
    assert manifest["processing"]["mode"] == "ml"
    assert manifest["source"] == {"file": "s.jsonl", "sha256": "sha256:abc"}
    assert manifest["counts"] == {"accepted": 1, "duplicate": 1, "rejected_lines": 1}
    statuses = [(e["line_no"], e["status"]) for e in manifest["entries"]]
    assert statuses == [(1, "accepted"), (2, "duplicate"), (3, "format")]
    accepted = manifest["entries"][0]
    assert accepted["content_hash"] == report.accepted[0].content_hash
    assert manifest["entries"][2]["error_codes"] == ["invalid_json"]


def test_manifest_is_reproducible(make_record):
    lines = [json.dumps(make_record(submission_id=f"s{i}")) for i in range(3)]
    config = ProcessingConfig()

    first = build_manifest(_import(lines), config, source_name="f", source_sha256="h")
    second = build_manifest(_import(lines), config, source_name="f", source_sha256="h")

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_manifest_has_no_raw_content_or_provenance_labels(make_record):
    record = make_record(text="UNIQUE-TEXT-MARKER", code="UNIQUE_CODE_MARKER = 1\n")
    manifest = build_manifest(_import([json.dumps(record)]), ProcessingConfig(), source_name="f", source_sha256="h")
    dumped = json.dumps(manifest)

    assert "UNIQUE-TEXT-MARKER" not in dumped
    assert "UNIQUE_CODE_MARKER" not in dumped
    # ai_declaration is a self-declaration, not a provenance label.
    assert "ai_declaration" not in dumped
    assert "label" not in dumped
    assert manifest["raw_content_fields"] == ["text", "code"]
    assert manifest["feature_sets"] == []
