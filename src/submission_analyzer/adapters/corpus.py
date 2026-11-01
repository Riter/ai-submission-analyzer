"""Load a labeled corpus directory (lab 2 layout) into a feature index and metadata."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from submission_analyzer.adapters.loaders import load_assignments, load_pairs, load_provenance, load_split
from submission_analyzer.adapters.sources import JsonlFileSource
from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.corpus import LabeledPair, Provenance, Split, check_no_family_leakage
from submission_analyzer.core.errors import DataSourceError
from submission_analyzer.core.hashing import file_sha256
from submission_analyzer.core.models import Assignment
from submission_analyzer.ingestion.service import ImportReport, IngestionService
from submission_analyzer.similarity.index import FeatureIndex
from submission_analyzer.similarity.pipeline import build_index, iter_accepted, iter_features
from submission_analyzer.storage.repository import InMemorySubmissionRepository

CORPUS_FILES = ("assignments.json", "submissions.jsonl", "provenance.jsonl", "pairs.jsonl", "split.json")


@dataclass(frozen=True, slots=True)
class Corpus:
    assignments: dict[str, Assignment]
    index: FeatureIndex
    pairs: list[LabeledPair]
    split: Split
    provenance: list[Provenance]
    import_report: ImportReport
    file_hashes: dict[str, str]
    directory: Path

    def template_blind_index(self, config: SimilarityConfig) -> FeatureIndex:
        """The same submissions with templates ignored (ablation of the template exclusion)."""
        blind = {aid: dataclasses.replace(a, allowed_template="") for aid, a in self.assignments.items()}
        return _stream_index(self.directory / "submissions.jsonl", blind, config, ImportReport())


def load_corpus(directory: Path, config: SimilarityConfig) -> Corpus:
    missing = [name for name in CORPUS_FILES if not (directory / name).is_file()]
    if missing:
        raise DataSourceError(f"{directory}: missing corpus files: {', '.join(missing)}")
    assignments = {a.assignment_id: a for a in load_assignments(directory / "assignments.json")}
    pairs = load_pairs(directory / "pairs.jsonl")
    split = load_split(directory / "split.json")
    provenance = load_provenance(directory / "provenance.jsonl")
    problems = check_no_family_leakage(split, pairs, provenance)
    if problems:
        raise DataSourceError(f"{directory}: family leakage between tune and eval: {'; '.join(problems[:3])}")

    report = ImportReport()
    index = _stream_index(directory / "submissions.jsonl", assignments, config, report)
    return Corpus(
        assignments=assignments,
        index=index,
        pairs=pairs,
        split=split,
        provenance=provenance,
        import_report=report,
        file_hashes={f"sha256:{name}": file_sha256(directory / name) for name in CORPUS_FILES},
        directory=directory,
    )


def _stream_index(
    path: Path, assignments: dict[str, Assignment], config: SimilarityConfig, report: ImportReport
) -> FeatureIndex:
    service = IngestionService(InMemorySubmissionRepository(), assignments.keys())
    with JsonlFileSource(path) as source:
        return build_index(iter_features(iter_accepted(service, source, report), assignments, config))
