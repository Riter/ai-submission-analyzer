"""Command-line interface.

The CLI is a thin adapter: it reads files, builds the service and prints JSON.
All validation happens in the domain layer, which can also be called directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from submission_analyzer import __version__
from submission_analyzer.adapters.loaders import load_assignments
from submission_analyzer.adapters.sources import JsonlFileSource
from submission_analyzer.analysis.baseline import BaselineProcessor
from submission_analyzer.analysis.registry import default_registry
from submission_analyzer.analysis.service import AnalysisService
from submission_analyzer.core.config import ProcessingConfig, load_config
from submission_analyzer.core.errors import ConfigError, DataSourceError
from submission_analyzer.core.hashing import file_sha256
from submission_analyzer.ingestion.manifest import build_manifest
from submission_analyzer.ingestion.service import ImportReport, IngestionService
from submission_analyzer.storage.repository import InMemorySubmissionRepository

EXIT_OK = 0
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="submission-analyzer", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("file", type=Path, help="JSONL file with one submission per line")
        p.add_argument("--assignments", type=Path, help="assignments.json; enables assignment_id check")
        p.add_argument("--config", type=Path, help="processing config (TOML); defaults to llm mode")

    p_import = sub.add_parser("import", help="validate and import submissions, print a JSON report")
    add_common(p_import)
    p_import.add_argument("--manifest", type=Path, help="write the corpus manifest to this path")

    p_baseline = sub.add_parser("baseline", help="import, then run the deterministic baseline on accepted submissions")
    add_common(p_baseline)

    p_analyze = sub.add_parser("analyze", help="import, then run the configured analyzers; print reports")
    add_common(p_analyze)
    p_analyze.set_defaults(assignments_required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "assignments_required", False) and args.assignments is None:
        parser.error(f"{args.command} requires --assignments")
    try:
        config = load_config(args.config) if args.config else ProcessingConfig()
        assignments = (
            {a.assignment_id: a for a in load_assignments(args.assignments)} if args.assignments else None
        )
        repository = InMemorySubmissionRepository()
        service = IngestionService(repository, assignments.keys() if assignments is not None else None)
        analysis = (
            AnalysisService(default_registry(), config, assignments, repository)
            if args.command == "analyze" and assignments is not None
            else None
        )
        report = _import_file(service, args.file)
    except (ConfigError, DataSourceError, OSError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.command == "import":
        output: dict[str, Any] = report.to_dict()
        if args.manifest:
            manifest = build_manifest(
                report, config, source_name=args.file.name, source_sha256=file_sha256(args.file)
            )
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output["manifest"] = str(args.manifest)
    elif analysis is not None:
        output = {
            "mode": str(config.mode),
            "config_hash": analysis.config_hash,
            "import": report.to_dict()["counts"],
            "reports": [r.to_dict() for r in analysis.analyze_all()],
        }
    else:
        processor = BaselineProcessor(config)
        output = {
            "mode": str(config.mode),
            "results": [processor.process(s).to_dict() for s in repository],
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return EXIT_OK


def _import_file(service: IngestionService, path: Path) -> ImportReport:
    # The source owns the file handle; the service only sees (line_no, raw) pairs.
    with JsonlFileSource(path) as source:
        return service.import_numbered(source)


if __name__ == "__main__":
    sys.exit(main())
