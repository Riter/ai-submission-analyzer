"""Wall time and peak memory of similarity processing for 100 / 1000 / 5000 records.

Usage:
    python scripts/profile_similarity.py [--sizes 100 1000 5000] [--repeats 3] [--seed 42] [--cprofile 1000]

Processing = read JSONL -> validate -> tokenize features -> index -> Jaccard for
every pair of one assignment. Variants:
  all-evidence     (before the optimization) every pair result is kept with line evidence;
  review-evidence  (after) pairs are streamed, evidence is built only for pairs with
                   score >= --threshold, and only those results are kept. Records above the
corpus size are synthetic copies with extra lines (load only, no quality claims),
spread over the 4 corpus assignments. Time and memory are measured in separate
runs because ``tracemalloc`` slows the code down. Each value is the median of
``--repeats`` runs.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from measure_index_memory import synthetic_lines  # same directory

from submission_analyzer.adapters.loaders import load_assignments
from submission_analyzer.adapters.sources import JsonlFileSource
from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.ingestion.service import ImportReport, IngestionService
from submission_analyzer.similarity.methods import JaccardMethod
from submission_analyzer.similarity.pairs import compare_all, iter_compare
from submission_analyzer.similarity.pipeline import build_index, iter_accepted, iter_features
from submission_analyzer.storage.repository import InMemorySubmissionRepository

CORPUS = Path("datasets/lab2")


def process(path: Path, assignments: dict, config: SimilarityConfig, variant: str, threshold: float) -> int:
    service = IngestionService(InMemorySubmissionRepository(), assignments.keys())
    with JsonlFileSource(path) as source:
        index = build_index(iter_features(iter_accepted(service, source, ImportReport()), assignments, config))
    if variant == "all-evidence":
        cross, same = compare_all(index, JaccardMethod())
        return len(cross) + len(same)
    compared = 0
    flagged = []
    for result in iter_compare(index, JaccardMethod(), threshold=threshold):
        compared += 1
        if result.review_required:
            flagged.append(result)
    return compared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cprofile", type=int, help="also print a cProfile summary for this size")
    parser.add_argument("--variant", choices=["all-evidence", "review-evidence"], default="review-evidence")
    parser.add_argument("--threshold", type=float, default=0.59, help="review threshold (from the experiment)")
    args = parser.parse_args()

    import random

    assignments = {a.assignment_id: a for a in load_assignments(CORPUS / "assignments.json")}
    base = [json.loads(line) for line in (CORPUS / "submissions.jsonl").read_text(encoding="utf-8").splitlines()]
    config = SimilarityConfig()
    print(
        f"python {sys.version.split()[0]}, {platform.platform()}, {platform.processor() or platform.machine()}; "
        f"seed={args.seed}, repeats={args.repeats}, ngram={config.ngram}, method=jaccard, "
        f"assignments={len(assignments)}, variant={args.variant}, threshold={args.threshold}"
    )
    print("records  pairs       wall_s(median)  runs                      peak_MiB(median)")
    with tempfile.TemporaryDirectory() as tmp:
        for size in args.sizes:
            path = Path(tmp) / f"s{size}.jsonl"
            path.write_text("\n".join(synthetic_lines(base, size, random.Random(args.seed))) + "\n", encoding="utf-8")

            times: list[float] = []
            pairs = 0
            for _ in range(args.repeats):
                start = time.perf_counter()
                pairs = process(path, assignments, config, args.variant, args.threshold)
                times.append(time.perf_counter() - start)

            peaks: list[float] = []
            for _ in range(args.repeats):
                tracemalloc.start()
                process(path, assignments, config, args.variant, args.threshold)
                peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
                tracemalloc.stop()

            runs = ", ".join(f"{t:.2f}" for t in times)
            print(f"{size:<8} {pairs:<11} {statistics.median(times):<15.3f} {runs:<25} {statistics.median(peaks):.1f}")

            if args.cprofile == size:
                profiler = cProfile.Profile()
                profiler.runcall(process, path, assignments, config, args.variant, args.threshold)
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(12)
                profile_text = stream.getvalue()
        if args.cprofile in args.sizes:
            print(f"\ncProfile, {args.cprofile} records, top functions by own time:")
            print(profile_text)


if __name__ == "__main__":
    main()
