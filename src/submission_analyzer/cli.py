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
from submission_analyzer.adapters.loaders import load_assignments, load_split
from submission_analyzer.adapters.sources import JsonlFileSource
from submission_analyzer.analysis.baseline import BaselineProcessor
from submission_analyzer.analysis.registry import default_registry
from submission_analyzer.analysis.service import AnalysisService
from submission_analyzer.core.config import (
    SIMILARITY_METHODS,
    ProcessingConfig,
    config_hash,
    load_config,
)
from submission_analyzer.core.corpus import SplitName
from submission_analyzer.core.errors import ConfigError, DataSourceError
from submission_analyzer.core.hashing import file_sha256
from submission_analyzer.core.models import Assignment
from submission_analyzer.ingestion.manifest import build_manifest
from submission_analyzer.ingestion.service import ImportReport, IngestionService
from submission_analyzer.similarity.methods import SimilarityError, create_method
from submission_analyzer.similarity.pairs import compare_all, top_k
from submission_analyzer.similarity.pipeline import (
    build_index,
    iter_accepted,
    iter_features,
)
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
        p.add_argument("--config", type=Path, help="processing config (TOML); defaults to ml mode")

    p_import = sub.add_parser("import", help="validate and import submissions, print a JSON report")
    add_common(p_import)
    p_import.add_argument("--manifest", type=Path, help="write the corpus manifest to this path")

    p_baseline = sub.add_parser("baseline", help="import, then run the deterministic baseline on accepted submissions")
    add_common(p_baseline)

    p_analyze = sub.add_parser("analyze", help="import, then run the configured analyzers; print reports")
    add_common(p_analyze)
    p_analyze.set_defaults(assignments_required=True)

    p_similarity = sub.add_parser(
        "similarity", help="stream submissions into a feature index and compare pairs of one assignment"
    )
    add_common(p_similarity)
    p_similarity.set_defaults(assignments_required=True)
    p_similarity.add_argument("--method", choices=SIMILARITY_METHODS, default="jaccard")
    p_similarity.add_argument("--split", type=Path, help="split.json; the method is fitted on the tune side only")
    p_similarity.add_argument("--submission", help="print top-k candidates for this submission instead of all pairs")
    p_similarity.add_argument("--top-k", type=int, help="number of candidates (default: [similarity].top_k)")
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
        if args.command == "similarity" and assignments is not None:
            print(json.dumps(_similarity(args, config, assignments, service), ensure_ascii=False, indent=2))
            return EXIT_OK
        analysis = (
            AnalysisService(default_registry(), config, assignments, repository)
            if args.command == "analyze" and assignments is not None
            else None
        )
        report = _import_file(service, args.file)
    except (ConfigError, DataSourceError, SimilarityError, OSError, UnicodeDecodeError) as exc:
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


def _similarity(
    args: argparse.Namespace,
    config: ProcessingConfig,
    assignments: dict[str, Assignment],
    service: IngestionService,
) -> dict[str, Any]:
    if args.method not in config.similarity.methods:
        raise ConfigError(f"method '{args.method}' is not enabled in [similarity].methods")
    if args.method == "tfidf" and args.split is None:
        raise ConfigError("--method tfidf requires --split: the vectorizer is fitted on the tune side only")
    if args.top_k is not None and args.top_k < 1:
        raise ConfigError("--top-k must be positive")

    report = ImportReport()
    # read -> validate -> features: one line at a time; only the index keeps the whole corpus.
    with JsonlFileSource(args.file) as source:
        index = build_index(iter_features(iter_accepted(service, source, report), assignments, config.similarity))

    method = create_method(args.method)
    fit: dict[str, Any] | None = None
    if args.split is not None:
        tune_ids = load_split(args.split).submission_ids(SplitName.TUNE)
        tune = [f for f in index if f.submission_id in tune_ids]
        method.fit(tune)
        fit = {"split": args.split.name, "side": str(SplitName.TUNE), "submissions": len(tune)}

    output: dict[str, Any] = {
        "config_hash": config_hash(config),
        "method": {"id": method.method_id, "version": method.method_version, "fit": fit},
        "import": report.to_dict()["counts"],
    }
    if args.submission is not None:
        if args.submission not in index:
            raise DataSourceError(f"submission '{args.submission}' is not among accepted submissions")
        k = args.top_k or config.similarity.top_k
        output["submission_id"] = args.submission
        output["top_k"] = [r.to_dict() for r in top_k(index, args.submission, method, k)]
    else:
        cross, same = compare_all(index, method)
        output["pairs"] = [r.to_dict() for r in cross]
        output["same_student_pairs"] = [r.to_dict() for r in same]
    return output


def _import_file(service: IngestionService, path: Path) -> ImportReport:
    # The source owns the file handle; the service only sees (line_no, raw) pairs.
    with JsonlFileSource(path) as source:
        return service.import_numbered(source)


if __name__ == "__main__":
    sys.exit(main())
