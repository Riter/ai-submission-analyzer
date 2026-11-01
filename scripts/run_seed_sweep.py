"""Show how unstable the metrics are on a small corpus: regenerate it with several seeds.

Usage:
    python scripts/run_seed_sweep.py [--seeds 1 2 3 4 5 42] [--mlflow mlruns]

For every seed the corpus is regenerated (different assignment of solutions to
family layouts, different edits in "modified" copies, different split of the
special families) and the experiment is run with that seed. The spread of eval
F1 across seeds shows how much a single number can be trusted.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from submission_analyzer.adapters.corpus import load_corpus
from submission_analyzer.core.config import ExperimentConfig, ProcessingConfig, load_config
from submission_analyzer.evaluation.experiment import run_experiment

GENERATOR = Path(__file__).with_name("generate_lab2_data.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 42])
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--mlflow", type=Path, help="also log every run to this MLflow store")
    args = parser.parse_args()

    base: ProcessingConfig = load_config(args.config)
    f1: dict[str, list[float | None]] = {}
    print(f"{'seed':<5} {'eval pairs':<11} {'detector':<34} {'F1 eval':>9} {'FN':>4} {'FP':>4}")
    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            out = Path(tmp) / f"seed-{seed}"
            command = [sys.executable, str(GENERATOR), "--seed", str(seed), "--out", str(out)]
            subprocess.run(command, check=True, capture_output=True)
            config = replace(base, experiment=ExperimentConfig(seed=seed, classifier=base.experiment.classifier))
            corpus = load_corpus(out, config.similarity)
            result = run_experiment(
                corpus.index,
                corpus.pairs,
                corpus.split,
                config,
                template_blind_index=corpus.template_blind_index(config.similarity),
                inputs={**corpus.file_hashes, "corpus_seed": str(seed)},
            )
            for report in result.detectors:
                matrix = report.eval.matrix
                f1.setdefault(report.detector_id, []).append(matrix.f1)
                print(
                    f"{seed:<5} {result.params['pairs_eval']:<11} {report.detector_id:<34} "
                    f"{matrix.f1 if matrix.f1 is not None else '-':>9} {matrix.fn:>4} {matrix.fp:>4}"
                )
            if args.mlflow is not None:
                from submission_analyzer.adapters.tracking import log_experiment

                log_experiment(result, args.mlflow, [out / "split.json", out / "pairs.jsonl"], run_name=f"sweep-{seed}")

    print("\ndetector                              F1 eval: min / median / max over seeds")
    for detector_id, values in f1.items():
        known = [v for v in values if v is not None]
        print(f"{detector_id:<37} {min(known):.3f} / {statistics.median(known):.3f} / {max(known):.3f}")


if __name__ == "__main__":
    main()
