"""Defense rehearsal: the teacher adds a new template — what changes and why."""

from __future__ import annotations

import json
import shutil

from submission_analyzer.adapters.corpus import load_corpus
from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.ingestion.validation import validate_record
from submission_analyzer.similarity.features import extract_features
from submission_analyzer.similarity.methods import JaccardMethod
from submission_analyzer.similarity.pairs import compare

from .conftest import LAB2_DIR, TEMPLATE

# Two students share a validation block that the teacher later adds to the template.
SHARED_BLOCK = "    if n < 0:\n        raise ValueError('n must be non-negative')\n"
NEW_TEMPLATE = TEMPLATE.replace("    result: list[str] = []\n", "    result: list[str] = []\n" + SHARED_BLOCK)
CODE_A = NEW_TEMPLATE.replace(
    "    return result", "    for i in range(n):\n        result.append(str(i))\n    return result"
)
CODE_B = NEW_TEMPLATE.replace("    return result", "    result.extend(map(str, range(n)))\n    return result")


def _pair(make_record, template: str):
    config = SimilarityConfig()
    a = validate_record(make_record(submission_id="a", student_id="p1", code=CODE_A))
    b = validate_record(make_record(submission_id="b", student_id="p2", code=CODE_B))
    features_a = extract_features(a, template, config)
    return compare(features_a, extract_features(b, template, config), JaccardMethod()), features_a


def test_new_template_moves_shared_block_out_of_the_score(make_record):
    before, features_before = _pair(make_record, TEMPLATE)
    after, features_after = _pair(make_record, NEW_TEMPLATE)

    assert after.score < before.score  # the shared block no longer counts as shared work
    assert after.shared_grams < before.shared_grams
    # ...its n-grams are now template n-grams, reported separately and excluded from the score.
    assert len(features_after.template_grams) > len(features_before.template_grams)
    assert after.template_shared == 1.0 and "template" in after.explanation


def test_new_template_changes_the_recorded_corpus_version(tmp_path):
    corpus_dir = tmp_path / "lab2"
    shutil.copytree(LAB2_DIR, corpus_dir)
    before = load_corpus(corpus_dir, SimilarityConfig())

    path = corpus_dir / "assignments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["assignments"][0]["allowed_template"] += "\n\ndef check(n: int) -> None:\n    assert n >= 0\n"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    after = load_corpus(corpus_dir, SimilarityConfig())

    # The experiment run records this hash, so results with the old and new template are told apart.
    assert after.file_hashes["sha256:assignments.json"] != before.file_hashes["sha256:assignments.json"]
    assert after.file_hashes["sha256:submissions.jsonl"] == before.file_hashes["sha256:submissions.jsonl"]
