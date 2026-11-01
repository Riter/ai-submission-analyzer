"""Measure the memory held by the similarity feature index.

Usage:
    python scripts/measure_index_memory.py [--sizes 40 1000] [--repeats 3] [--seed 42]

Synthetic records for sizes above the corpus are copies of corpus submissions
with new ids and students and a few random extra lines; they only create load
and say nothing about quality. Memory is measured with ``tracemalloc``:
``index_kib`` is what the index keeps after the stream is consumed (submissions
already dropped), ``peak_kib`` is the peak while building it.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import platform
import random
import statistics
import sys
import tracemalloc
from pathlib import Path

from submission_analyzer.adapters.loaders import load_assignments
from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.models import Submission
from submission_analyzer.ingestion.service import ImportReport, IngestionService
from submission_analyzer.similarity.pipeline import build_index, iter_accepted, iter_features
from submission_analyzer.storage.repository import InMemorySubmissionRepository

CORPUS = Path("datasets/lab2")


def synthetic_lines(base: list[dict], size: int, rng: random.Random) -> list[str]:
    lines: list[str] = []
    for i in range(size):
        record = dict(base[i % len(base)])
        if i >= len(base):
            record["submission_id"] = f"syn-{i:05d}"
            record["student_id"] = f"syn-student-{i:05d}"
            extra = "".join(f"\nvalue_{rng.randint(0, 999)} = {rng.randint(0, 99)}" for _ in range(rng.randint(0, 5)))
            record["code"] = record["code"] + extra + "\n"
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


class _DropAfterValidation(InMemorySubmissionRepository):
    """Remembers only ids and hashes, so the measurement shows the index alone."""

    def add(self, submission: Submission) -> None:
        super().add(dataclasses.replace(submission, text="", code=""))


def measure(lines: list[str], assignments: dict, config: SimilarityConfig) -> tuple[float, float, int, int]:
    gc.collect()
    tracemalloc.start()
    service = IngestionService(_DropAfterValidation(), assignments.keys())
    index = build_index(iter_features(iter_accepted(service, enumerate(lines, 1), ImportReport()), assignments, config))
    del service
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    grams = sum(len(f.grams) + len(f.template_grams) for f in index)
    return current / 1024, peak / 1024, len(index), grams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assignments = {a.assignment_id: a for a in load_assignments(CORPUS / "assignments.json")}
    base = [json.loads(line) for line in (CORPUS / "submissions.jsonl").read_text(encoding="utf-8").splitlines()]
    config = SimilarityConfig()
    print(f"python {sys.version.split()[0]}, {platform.platform()}, seed={args.seed}, ngram={config.ngram}")
    print("size  indexed  grams     index_KiB(median)  peak_KiB(median)  KiB/submission")
    for size in args.sizes:
        lines = synthetic_lines(base, size, random.Random(args.seed))
        runs = [measure(lines, assignments, config) for _ in range(args.repeats)]
        index_kib = statistics.median(r[0] for r in runs)
        peak_kib = statistics.median(r[1] for r in runs)
        indexed, grams = runs[0][2], runs[0][3]
        print(f"{size:<5} {indexed:<8} {grams:<9} {index_kib:<18.1f} {peak_kib:<17.1f} {index_kib / indexed:.2f}")


if __name__ == "__main__":
    main()
