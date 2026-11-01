"""The evaluation experiment on the lab 2 corpus: no leakage, reproducibility, MLflow run."""

from __future__ import annotations

import dataclasses

import pytest

from submission_analyzer.adapters.corpus import load_corpus
from submission_analyzer.core.config import ProcessingConfig
from submission_analyzer.core.corpus import DerivationRelation, Split, SplitName
from submission_analyzer.evaluation.classifier import vectorize
from submission_analyzer.evaluation.dataset import ExperimentError, build_pair_rows, by_split, fit_methods
from submission_analyzer.evaluation.experiment import TEMPLATE_BLIND_ID, run_experiment

from .conftest import LAB2_DIR

CONFIG = ProcessingConfig()


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(LAB2_DIR, CONFIG.similarity)


@pytest.fixture(scope="module")
def result(corpus):
    return run_experiment(
        corpus.index,
        corpus.pairs,
        corpus.split,
        CONFIG,
        template_blind_index=corpus.template_blind_index(CONFIG.similarity),
        inputs=corpus.file_hashes,
    )


def test_compares_baselines_and_classifier_on_the_same_eval_set(result):
    ids = [d.detector_id for d in result.detectors]
    assert ids == ["threshold:jaccard", "threshold:tfidf_cosine", "classifier:logistic_regression", TEMPLATE_BLIND_ID]
    totals = {d.eval.total for d in result.detectors}
    assert totals == {result.params["pairs_eval"]}
    for report in result.detectors:
        assert report.eval.matrix.f1 is not None


def test_errors_are_real_misclassifications_on_eval(result):
    assert result.errors
    for error in result.errors:
        assert error.pair.split is SplitName.EVAL
        derived = error.pair.derivation_relation is not DerivationRelation.INDEPENDENT
        assert (error.kind == "false_negative") == derived


def test_template_only_pairs_stay_below_the_threshold(result):
    """The allowed template alone does not trigger review_required on this corpus."""
    for detector_id in ("threshold:jaccard", "threshold:tfidf_cosine"):
        report = result.detector(detector_id)
        assert report.template_only_max < report.threshold
    assert not any(e.pair.template_only and e.kind == "false_positive" for e in result.errors)


def test_experiment_is_reproducible(corpus, result):
    again = run_experiment(
        corpus.index,
        corpus.pairs,
        corpus.split,
        CONFIG,
        template_blind_index=corpus.template_blind_index(CONFIG.similarity),
        inputs=corpus.file_hashes,
    )
    assert again.to_dict() == result.to_dict()


def test_eval_labels_do_not_influence_anything_fitted(corpus, result):
    """Flipping every eval label changes eval metrics but no threshold, model weight or tune metric."""

    def flip(pair):
        if pair.split is not SplitName.EVAL:
            return pair
        relation = DerivationRelation.INDEPENDENT if pair.is_derived else DerivationRelation.MODIFIED
        return dataclasses.replace(pair, derivation_relation=relation)

    flipped = run_experiment(corpus.index, [flip(p) for p in corpus.pairs], corpus.split, CONFIG)
    for report in flipped.detectors:
        original = result.detector(report.detector_id)
        assert report.threshold == original.threshold
        assert report.tune == original.tune
    original_model, flipped_model = result.classifier.estimator[-1], flipped.classifier.estimator[-1]
    assert (original_model.coef_ == flipped_model.coef_).all()


def test_tfidf_vocabulary_comes_from_tune_submissions_only(corpus):
    methods = fit_methods(corpus.index, corpus.split, CONFIG.similarity)
    tfidf = next(m for m in methods if m.method_id == "tfidf")
    tune_ids = corpus.split.submission_ids(SplitName.TUNE)
    tune_terms = {" ".join(g) for f in corpus.index if f.submission_id in tune_ids for g in f.grams}
    eval_only = {" ".join(g) for f in corpus.index if f.submission_id not in tune_ids for g in f.grams} - tune_terms
    assert set(tfidf.vocabulary) <= tune_terms
    assert eval_only and not (eval_only & set(tfidf.vocabulary))


def test_family_leakage_stops_the_experiment(corpus):
    pair = next(p for p in corpus.pairs if p.split is SplitName.EVAL)
    families = dict(corpus.split.families)
    families[pair.family_a] = SplitName.TUNE  # one family now sits on both sides
    with pytest.raises(ExperimentError, match="leakage"):
        run_experiment(corpus.index, corpus.pairs, Split(families, corpus.split.submissions), CONFIG)


def test_incompatible_feature_schema_stops_the_experiment(corpus):
    with pytest.raises(ExperimentError, match="pair-1"):
        run_experiment(corpus.index, corpus.pairs, corpus.split, ProcessingConfig(feature_schema_version="pair-0"))


def test_params_record_inputs_and_configuration(result):
    params = result.params
    assert params["seed"] == 42 and params["feature_schema_version"] == "pair-1"
    assert params["config_hash"].startswith("sha256:")
    assert {"sha256:split.json", "sha256:pairs.jsonl", "sha256:assignments.json"} <= set(params)
    assert params["pairs_tune"] + params["pairs_eval"] == 110


# --- MLflow: the saved experiment can be reproduced ---


@pytest.fixture(scope="module")
def logged(result, tmp_path_factory):
    from submission_analyzer.adapters.tracking import log_experiment

    store = tmp_path_factory.mktemp("mlflow")
    run = log_experiment(result, store, [LAB2_DIR / "split.json", LAB2_DIR / "pairs.jsonl"], run_name="test")
    return store, run


def test_mlflow_run_stores_params_metrics_and_artifacts(logged, result):
    from mlflow.tracking import MlflowClient

    from submission_analyzer.adapters.tracking import load_run

    store, run = logged
    params, metrics, _ = load_run(store, run.run_id)
    assert params["config_hash"] == result.params["config_hash"]
    assert params["sha256:split.json"] == result.params["sha256:split.json"]
    assert metrics["eval/threshold:jaccard/f1"] == result.detector("threshold:jaccard").eval.matrix.f1
    client = MlflowClient(run.tracking_uri)
    names = {a.path for a in client.list_artifacts(run.run_id)} | {
        a.path for a in client.list_artifacts(run.run_id, "data")
    }
    assert {"experiment.json", "data/split.json", "data/pairs.jsonl"} <= names


def test_logged_model_reproduces_predictions(logged, result, corpus):
    from submission_analyzer.adapters.tracking import load_run

    store, run = logged
    _, _, model = load_run(store, run.run_id)
    methods = fit_methods(corpus.index, corpus.split, CONFIG.similarity)
    rows = by_split(build_pair_rows(corpus.index, corpus.pairs, methods), SplitName.EVAL)
    vectors = [vectorize(r.features) for r in rows]
    assert (model.predict_proba(vectors) == result.classifier.estimator.predict_proba(vectors)).all()
