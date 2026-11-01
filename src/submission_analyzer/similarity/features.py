"""Code features for similarity: token n-grams built with :mod:`tokenize`.

Student code is only tokenized (and parsed by :func:`ast.parse` to flag a
syntax error) — it is never executed or imported.
"""

from __future__ import annotations

import ast
import builtins
import io
import keyword
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass

from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.models import Submission
from submission_analyzer.core.textutils import source_lines

Gram = tuple[str, ...]
LineSpan = tuple[int, int]

# Layout and comments carry no meaning for similarity.
_SKIPPED = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.TYPE_COMMENT,
    }
)
_KEPT_NAMES = frozenset(keyword.kwlist) | frozenset(keyword.softkwlist) | frozenset(dir(builtins))
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)  # Python 3.12+


@dataclass(frozen=True, slots=True)
class Token:
    value: str  # normalized value used in n-grams
    start_line: int
    end_line: int
    in_template: bool


@dataclass(frozen=True, slots=True)
class CodeFeatures:
    """Features of one submission's code. Raw ``text``/``code`` is not kept.

    ``grams`` maps each n-gram of the student's own code to the line spans of
    its occurrences. N-grams made only of allowed-template lines are kept apart
    in ``template_grams`` so that the template never counts as shared work.
    """

    submission_id: str
    assignment_id: str
    student_id: str
    content_hash: str
    line_count: int
    token_count: int
    grams: Mapping[Gram, tuple[LineSpan, ...]]
    template_grams: frozenset[Gram]
    parse_error: str | None = None
    syntax_error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.grams


def _meaningful_tokens(code: str) -> list[tokenize.TokenInfo]:
    return [tok for tok in tokenize.generate_tokens(io.StringIO(code).readline) if tok.type not in _SKIPPED]


def _line_signatures(tokens: list[tokenize.TokenInfo]) -> dict[int, tuple[str, ...]]:
    """Raw tokens of each line: layout and comments do not change a line's signature."""
    lines: dict[int, list[str]] = {}
    for tok in tokens:
        lines.setdefault(tok.start[0], []).append(tok.string)
    return {no: tuple(values) for no, values in lines.items()}


def template_signatures(template: str) -> frozenset[tuple[str, ...]]:
    """Token signatures of the allowed template's lines (empty if the template does not tokenize)."""
    try:
        return frozenset(_line_signatures(_meaningful_tokens(template)).values())
    except (tokenize.TokenError, SyntaxError):
        return frozenset()


def tokenize_code(code: str, template: frozenset[tuple[str, ...]], *, normalize: bool) -> list[Token]:
    """Return meaningful tokens of ``code``.

    A token belongs to the template when its line has exactly the tokens of a
    template line, so re-indenting or commenting a template line keeps it a
    template line.

    Raises:
        tokenize.TokenError, IndentationError, SyntaxError: the code cannot be tokenized.
    """
    raw = _meaningful_tokens(code)
    template_rows = {no for no, sig in _line_signatures(raw).items() if sig in template}
    tokens: list[Token] = []
    previous: tokenize.TokenInfo | None = None
    for tok in raw:
        value = _normalize(tok, previous) if normalize else tok.string
        tokens.append(Token(value, tok.start[0], tok.end[0], tok.start[0] in template_rows))
        previous = tok
    return tokens


def _normalize(tok: tokenize.TokenInfo, previous: tokenize.TokenInfo | None) -> str:
    if tok.type == tokenize.NAME:
        # Attribute names (``x.append``) belong to the API being used, not to the student.
        after_dot = previous is not None and previous.type == tokenize.OP and previous.string == "."
        return tok.string if tok.string in _KEPT_NAMES or after_dot else "ID"
    if tok.type == tokenize.STRING or tok.type == _FSTRING_MIDDLE:
        return "STR"
    return tok.string


def build_grams(tokens: list[Token], n: int) -> tuple[dict[Gram, tuple[LineSpan, ...]], frozenset[Gram]]:
    own: dict[Gram, list[LineSpan]] = {}
    template: set[Gram] = set()
    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        gram = tuple(t.value for t in window)
        if all(t.in_template for t in window):
            template.add(gram)
        else:
            own.setdefault(gram, []).append((window[0].start_line, window[-1].end_line))
    return {g: tuple(spans) for g, spans in own.items()}, frozenset(template)


def extract_features(submission: Submission, template: str, config: SimilarityConfig) -> CodeFeatures:
    """Build features of ``submission.code``; bad code yields empty features with a reason."""
    code = submission.code
    parse_error: str | None = None
    try:
        tokens = tokenize_code(code, template_signatures(template), normalize=config.normalize_identifiers)
    except (tokenize.TokenError, SyntaxError) as exc:  # IndentationError is a SyntaxError
        tokens = []
        parse_error = f"{type(exc).__name__}: {_error_message(exc)}"

    syntax_error: str | None = None
    try:
        ast.parse(code)
    except SyntaxError as exc:
        syntax_error = f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})"
    except ValueError as exc:  # e.g. null bytes in source
        syntax_error = f"ValueError: {exc}"

    grams, template_grams = build_grams(tokens, config.ngram)
    return CodeFeatures(
        submission_id=submission.submission_id,
        assignment_id=submission.assignment_id,
        student_id=submission.student_id,
        content_hash=submission.content_hash,
        line_count=len(source_lines(code)),
        token_count=len(tokens),
        grams=grams,
        template_grams=template_grams,
        parse_error=parse_error,
        syntax_error=syntax_error,
    )


def _error_message(exc: Exception) -> str:
    if isinstance(exc, SyntaxError):
        return f"{exc.msg} (line {exc.lineno})"
    return str(exc.args[0]) if exc.args else str(exc)
