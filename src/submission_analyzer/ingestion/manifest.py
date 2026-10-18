"""Corpus manifest: what was imported, from which file, by which processing version.

The manifest lists identifiers and hashes only. It holds neither raw content
(that stays in the submissions) nor labels; in particular ``ai_declaration`` is
not recorded as a provenance label.
"""

from __future__ import annotations

from typing import Any

from submission_analyzer.core.config import ProcessingConfig, config_hash
from submission_analyzer.ingestion.service import ImportReport

MANIFEST_SCHEMA_VERSION = "1"
RAW_CONTENT_FIELDS: tuple[str, ...] = ("text", "code")


def build_manifest(
    report: ImportReport,
    config: ProcessingConfig,
    *,
    source_name: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Build a manifest dict. It has no wall-clock fields, so re-running the
    same import with the same config produces an identical manifest."""
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "processing": {
            "mode": str(config.mode),
            "processing_version": config.processing_version,
            "prompt_template_version": config.prompt_template_version,
            "feature_schema_version": config.feature_schema_version,
            "analyzers": list(config.analyzers),
            "config_hash": config_hash(config),
        },
        "source": {"file": source_name, "sha256": source_sha256},
        "raw_content_fields": list(RAW_CONTENT_FIELDS),
        # Features are computed in later labs and stored separately, keyed by content_hash.
        "feature_sets": [],
        "counts": {
            "accepted": len(report.accepted),
            "duplicate": len(report.duplicate),
            "rejected_lines": len(report.errors),
        },
        "entries": [
            {
                "line_no": r.line_no,
                "status": str(r.status),
                "submission_id": r.submission_id,
                "content_hash": r.content_hash,
                "error_codes": [d.code for d in r.errors],
            }
            for r in report.lines
        ],
    }
