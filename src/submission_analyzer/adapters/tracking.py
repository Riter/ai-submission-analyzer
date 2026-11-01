"""MLflow adapter: stores experiment parameters, metrics, split artifacts and the model.

The tracking store is a local SQLite database with artifacts next to it
(``<store>/mlflow.db``, ``<store>/artifacts``); nothing leaves the machine.
The domain (``evaluation``) does not import MLflow.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from submission_analyzer.evaluation.experiment import ExperimentResult

EXPERIMENT_NAME = "lab2-review-required"


@dataclass(frozen=True, slots=True)
class LoggedRun:
    run_id: str
    model_uri: str
    tracking_uri: str


def tracking_uri(store: Path) -> str:
    return f"sqlite:///{store.resolve() / 'mlflow.db'}"


def _mlflow(store: Path) -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(store))
    return mlflow


def log_experiment(
    result: ExperimentResult,
    store: Path,
    artifacts: Sequence[Path],
    *,
    run_name: str | None = None,
) -> LoggedRun:
    """Log one experiment run; ``artifacts`` (split, pairs, ...) are stored under ``data/``."""
    mlflow = _mlflow(store)
    import mlflow.sklearn as mlflow_sklearn

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(store.resolve() / "artifacts").as_uri())
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params({key: str(value) for key, value in result.params.items()})
        mlflow.log_metrics(result.metrics())
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path="data")
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "experiment.json"
            report.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            mlflow.log_artifact(str(report))
        info = mlflow_sklearn.log_model(result.classifier.estimator, name="model")
        mlflow.set_tag("feature_schema_version", result.classifier.feature_schema)
        mlflow.set_tag("model_uri", info.model_uri)
    return LoggedRun(run.info.run_id, info.model_uri, tracking_uri(store))


def load_run(store: Path, run_id: str) -> tuple[dict[str, str], dict[str, float], Any]:
    """Parameters, metrics and the logged estimator of a run (for reproducibility checks)."""
    mlflow = _mlflow(store)
    import mlflow.sklearn as mlflow_sklearn

    run = mlflow.get_run(run_id)
    model = mlflow_sklearn.load_model(run.data.tags["model_uri"])
    return dict(run.data.params), dict(run.data.metrics), model
