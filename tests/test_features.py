"""Token features: what counts, what is dropped, how the template is kept apart."""

from __future__ import annotations

import pytest

from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.textutils import source_lines
from submission_analyzer.ingestion.validation import validate_record
from submission_analyzer.similarity.features import extract_features, template_signatures, tokenize_code

from .conftest import TEMPLATE

SOLUTION = (
    "def fizzbuzz(n: int) -> list[str]:\n"
    "    result: list[str] = []\n"
    "    for i in range(1, n + 1):\n"
    "        result.append('Fizz' if i % 3 == 0 else str(i))\n"
    "    return result\n"
)


@pytest.fixture
def features(make_record):
    def factory(code: str, template: str = TEMPLATE, **config):
        submission = validate_record(make_record(code=code))
        return extract_features(submission, template, SimilarityConfig(**config))

    return factory


def values(code: str, *, normalize: bool = True) -> list[str]:
    return [t.value for t in tokenize_code(code, frozenset(), normalize=normalize)]


def test_comments_and_layout_tokens_are_dropped():
    assert values("x = 1  # comment\n\n\n# another\nif x:\n    y = 2\n") == [
        "ID", "=", "1", "if", "ID", ":", "ID", "=", "2",
    ]  # fmt: skip


def test_identifiers_and_strings_are_normalized_but_keywords_and_builtins_kept():
    assert values("total = len('abc')\nitems.append(total)\n") == [
        "ID", "=", "len", "(", "STR", ")", "ID", ".", "append", "(", "ID", ")",
    ]  # fmt: skip
    assert values("total = len('abc')\n", normalize=False) == ["total", "=", "len", "(", "'abc'", ")"]


def test_comments_and_layout_do_not_change_features(features):
    reformatted = "# header\n\n" + SOLUTION.replace("    ", "  ").replace("[]\n", "[]  # init\n", 1)
    assert features(reformatted).grams.keys() == features(SOLUTION).grams.keys()


def test_template_grams_are_kept_apart(features):
    f = features(SOLUTION)
    assert f.template_grams
    # Lines 1, 2 and 5 are template lines; every own n-gram touches line 3 or 4.
    own_lines = {3, 4}
    assert all(
        own_lines & set(range(start, end + 1)) for occurrences in f.grams.values() for start, end in occurrences
    )
    only_template = features(TEMPLATE)
    assert only_template.is_empty and only_template.template_grams


def test_commented_template_line_still_counts_as_template():
    template = template_signatures(TEMPLATE)
    tokens = tokenize_code("def fizzbuzz(n: int) -> list[str]:\n  return result  # done\n", template, normalize=True)
    assert all(t.in_template for t in tokens)


def test_gram_line_spans_point_to_source_lines(features):
    f = features("\n\n" + SOLUTION)
    total = len(source_lines("\n\n" + SOLUTION))
    spans = [span for occurrences in f.grams.values() for span in occurrences]
    assert spans and all(1 <= start <= end <= total for start, end in spans)
    assert min(start for start, _ in spans) >= 3  # two leading blank lines are counted


@pytest.mark.parametrize("code", ["# only a comment\n", "\n\n", "x\n"])
def test_empty_features(features, code):
    f = features(code, template="")
    assert f.is_empty and f.parse_error is None


def test_untokenizable_code_gives_parse_error_not_exception(features):
    f = features("def f():\n    s = '''unterminated\n")
    assert f.is_empty
    assert f.parse_error and f.parse_error.startswith("TokenError")


def test_syntax_error_is_flagged_but_tokens_are_kept(features):
    f = features("def f(x)\n    return x + 1\n", template="")
    assert f.syntax_error and "SyntaxError" in f.syntax_error
    assert f.parse_error is None and not f.is_empty


def test_code_is_never_executed(features, tmp_path):
    marker = tmp_path / "executed"
    code = f"open({str(marker)!r}, 'w').write('x')\nimport os\nos.remove('/nonexistent')\n"
    features(code, template="")
    assert not marker.exists()


def test_ngram_size_is_configurable(features):
    assert len(next(iter(features(SOLUTION, ngram=2).grams))) == 2
    assert len(next(iter(features(SOLUTION, ngram=4).grams))) == 4
