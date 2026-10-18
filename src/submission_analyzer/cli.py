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
from submission_analyzer.baseline import BaselineProcessor
from submission_analyzer.config import ProcessingConfig, load_config
from submission_analyzer.errors import ConfigError, DataSourceError
from submission_analyzer.hashing import file_sha256
from submission_analyzer.loaders import load_assignments
from submission_analyzer.manifest import build_manifest
from submission_analyzer.repository import InMemorySubmissionRepository
from submission_analyzer.service import ImportReport, IngestionService

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config) if args.config else ProcessingConfig()
        known_ids = (
            {a.assignment_id for a in load_assignments(args.assignments)} if args.assignments else None
        )
        repository = InMemorySubmissionRepository()
        service = IngestionService(repository, known_ids)
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
    else:
        processor = BaselineProcessor(config)
        output = {
            "mode": str(config.mode),
            "results": [processor.process(s).to_dict() for s in repository],
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return EXIT_OK


def _import_file(service: IngestionService, path: Path) -> ImportReport:
    with path.open(encoding="utf-8") as fh:
        return service.import_lines(fh)


if __name__ == "__main__":
    sys.exit(main())
