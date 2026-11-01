"""Generate the synthetic lab 2 similarity corpus deterministically.

Usage:
    python scripts/generate_lab2_data.py [--out datasets/lab2] [--seed 42]

Produces:
    assignments.json   4 assignments, each with an allowed code template
    submissions.jsonl  submissions in the lab 1 ``Submission`` format
    provenance.jsonl   how each example was made: family_id, origin, derived_from
    pairs.jsonl        labeled pairs: derivation_relation, template_only, split
    split.json         tune/eval side of every family and submission

Every original solution starts a *family*; exact copies, reformatted and
renamed variants belong to the family of their original. Whole families are
assigned to the tune or eval side, so derivatives of one work never end up on
both sides. Labels describe how a synthetic example was made; they are not a
verdict about plagiarism. Re-running with the same seed gives identical files.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import random
import tokenize
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

MSK = timezone(timedelta(hours=3))
BASE_TIME = datetime(2026, 9, 28, 10, 0, tzinfo=MSK)

MIN_SUBMISSIONS = 30
MIN_PAIRS = 40
MIN_TEMPLATE_PAIRS = 5


def _main_block(call: str) -> str:
    return f'\n\nif __name__ == "__main__":\n{call}'


ASSIGNMENTS: list[dict[str, Any]] = [
    {
        "assignment_id": "lab-01-fizzbuzz",
        "title": "FizzBuzz",
        "requirements": [
            "Function fizzbuzz(n) returns a list of strings for 1..n",
            "Multiples of 3 are 'Fizz', of 5 are 'Buzz', of both are 'FizzBuzz'",
            "Report explains the approach",
        ],
        "allowed_template": "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n    return result\n"
        + _main_block("    for line in fizzbuzz(15):\n        print(line)\n"),
    },
    {
        "assignment_id": "lab-02-word-count",
        "title": "Word frequency",
        "requirements": [
            "Function word_count(text) returns a dict word -> count",
            "Words are compared case-insensitively",
            "Report describes how punctuation is handled",
        ],
        "allowed_template": "def word_count(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n"
        "    return counts\n" + _main_block('    print(word_count("To be, or not to be"))\n'),
    },
    {
        "assignment_id": "lab-03-transpose",
        "title": "Matrix transpose",
        "requirements": [
            "Function transpose(matrix) returns the transposed matrix",
            "Empty matrix returns an empty list",
            "Report states the time complexity",
        ],
        "allowed_template": "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    return []\n"
        + _main_block("    print(transpose([[1, 2, 3], [4, 5, 6]]))\n"),
    },
    {
        "assignment_id": "lab-04-palindrome",
        "title": "Palindrome check",
        "requirements": [
            "Function is_palindrome(s) returns True for palindromes",
            "Case and non-alphanumeric characters are ignored",
            "Report lists the tested edge cases",
        ],
        "allowed_template": "def is_palindrome(s: str) -> bool:\n    return False\n"
        + _main_block('    print(is_palindrome("A man, a plan, a canal: Panama"))\n'),
    },
]

_MAIN = {
    "lab-01-fizzbuzz": _main_block("    for line in fizzbuzz(15):\n        print(line)\n"),
    "lab-02-word-count": _main_block('    print(word_count("To be, or not to be"))\n'),
    "lab-03-transpose": _main_block("    print(transpose([[1, 2, 3], [4, 5, 6]]))\n"),
    "lab-04-palindrome": _main_block('    print(is_palindrome("A man, a plan, a canal: Panama"))\n'),
}

# Independent hand-written solutions: (body, report text, rename map used for the "renamed" variant).
# Every body is followed by the template's main block.
SOLUTIONS: dict[str, list[tuple[str, str, dict[str, str]]]] = {
    "lab-01-fizzbuzz": [
        (
            "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n            result.append(\"FizzBuzz\")\n        elif i % 3 == 0:\n"
            "            result.append(\"Fizz\")\n        elif i % 5 == 0:\n            result.append(\"Buzz\")\n"
            "        else:\n            result.append(str(i))\n    return result\n",
            "Перебираю числа от 1 до n и проверяю делимость сначала на 15, затем на 3 и 5.",
            {"i": "number"},
        ),
        (
            "def fizzbuzz(n: int) -> list[str]:\n    out = []\n    for k in range(1, n + 1):\n"
            "        word = (\"Fizz\" if k % 3 == 0 else \"\") + (\"Buzz\" if k % 5 == 0 else \"\")\n"
            "        out.append(word or str(k))\n    return out\n",
            "Строю слово конкатенацией: Fizz для кратных 3 и Buzz для кратных 5; пустое слово заменяю числом.",
            {"out": "words", "k": "idx", "word": "label"},
        ),
        (
            "def fizzbuzz(n: int) -> list[str]:\n"
            "    return [\"FizzBuzz\" if i % 15 == 0 else \"Fizz\" if i % 3 == 0 else \"Buzz\" if i % 5 == 0 else str(i)\n"
            "            for i in range(1, n + 1)]\n",
            "Решение одним списковым включением с цепочкой условных выражений.",
            {"i": "x"},
        ),
        (
            "RULES = ((15, \"FizzBuzz\"), (3, \"Fizz\"), (5, \"Buzz\"))\n\n\ndef label(value: int) -> str:\n"
            "    for divisor, name in RULES:\n        if value % divisor == 0:\n            return name\n"
            "    return str(value)\n\n\ndef fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n"
            "    for value in range(1, n + 1):\n        result.append(label(value))\n    return result\n",
            "Правила хранятся в таблице делителей; отдельная функция подбирает слово для числа.",
            {"RULES": "TABLE", "label": "word_for", "value": "num", "divisor": "d", "name": "text"},
        ),
    ],
    "lab-02-word-count": [
        (
            "import re\n\n\ndef word_count(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n"
            "    for word in re.findall(r\"[\\w']+\", text.lower()):\n        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n",
            "Привожу текст к нижнему регистру и выделяю слова регулярным выражением; пунктуация отбрасывается.",
            {"word": "token"},
        ),
        (
            "from collections import Counter\nimport string\n\n\ndef word_count(text: str) -> dict[str, int]:\n"
            "    cleaned = text.lower().translate(str.maketrans(\"\", \"\", string.punctuation))\n"
            "    return dict(Counter(cleaned.split()))\n",
            "Удаляю знаки пунктуации через str.translate и считаю слова с помощью Counter.",
            {"cleaned": "plain"},
        ),
        (
            "import string\n\n\ndef word_count(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n"
            "    for token in text.split():\n        word = token.strip(string.punctuation).lower()\n"
            "        if word:\n            counts[word] = counts.get(word, 0) + 1\n    return counts\n",
            "Разбиваю по пробелам и обрезаю пунктуацию по краям каждого слова.",
            {"token": "piece", "word": "key"},
        ),
        (
            "from collections import defaultdict\n\n\ndef word_count(text: str) -> dict[str, int]:\n"
            "    cleaned = \"\".join(ch.lower() if ch.isalnum() or ch.isspace() else \" \" for ch in text)\n"
            "    tally = defaultdict(int)\n    for word in cleaned.split():\n        tally[word] += 1\n"
            "    return dict(tally)\n",
            "Заменяю все небуквенные символы пробелами и считаю слова в defaultdict.",
            {"cleaned": "normalized", "ch": "c", "tally": "freq", "word": "w"},
        ),
    ],
    "lab-03-transpose": [
        (
            "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    if not matrix:\n        return []\n"
            "    return [list(row) for row in zip(*matrix)]\n",
            "Использую zip(*matrix). Сложность O(n*m), где n и m — размеры матрицы.",
            {"row": "column"},
        ),
        (
            "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    rows = len(matrix)\n"
            "    cols = len(matrix[0]) if rows else 0\n"
            "    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]\n",
            "Двойной цикл по индексам. Время O(rows*cols), дополнительная память того же порядка.",
            {"rows": "height", "cols": "width", "r": "i", "c": "j"},
        ),
        (
            "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    if not matrix:\n        return []\n"
            "    result = []\n    for col in range(len(matrix[0])):\n        new_row = []\n"
            "        for row in matrix:\n            new_row.append(row[col])\n        result.append(new_row)\n"
            "    return result\n",
            "Собираю каждую строку результата из столбца исходной матрицы. Сложность O(n*m).",
            {"result": "out", "col": "j", "new_row": "line", "row": "src"},
        ),
        (
            "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n"
            "    return list(map(list, zip(*matrix))) if matrix else []\n",
            "map(list, zip(*matrix)) в одну строку; пустая матрица обрабатывается отдельно. O(n*m).",
            {},
        ),
    ],
    "lab-04-palindrome": [
        (
            "def is_palindrome(s: str) -> bool:\n    cleaned = [ch.lower() for ch in s if ch.isalnum()]\n"
            "    return cleaned == cleaned[::-1]\n",
            "Оставляю только буквы и цифры и сравниваю список с перевёрнутым. Проверены пустая строка и один символ.",
            {"cleaned": "letters", "ch": "c"},
        ),
        (
            "def is_palindrome(s: str) -> bool:\n    left, right = 0, len(s) - 1\n    while left < right:\n"
            "        if not s[left].isalnum():\n            left += 1\n        elif not s[right].isalnum():\n"
            "            right -= 1\n        elif s[left].lower() != s[right].lower():\n            return False\n"
            "        else:\n            left += 1\n            right -= 1\n    return True\n",
            "Два указателя без копирования строки. Проверены строки из одной пунктуации и разный регистр.",
            {"left": "i", "right": "j"},
        ),
        (
            "import re\n\n\ndef is_palindrome(s: str) -> bool:\n"
            "    normalized = re.sub(r\"[^a-z0-9]\", \"\", s.lower())\n"
            "    return normalized == \"\".join(reversed(normalized))\n",
            "Регулярным выражением убираю всё, кроме букв и цифр. Проверены пустая строка и пробелы.",
            {"normalized": "core"},
        ),
        (
            "def is_palindrome(s: str) -> bool:\n    chars = \"\".join(filter(str.isalnum, s)).lower()\n\n"
            "    def check(lo: int, hi: int) -> bool:\n        if lo >= hi:\n            return True\n"
            "        return chars[lo] == chars[hi] and check(lo + 1, hi - 1)\n\n    return check(0, len(chars) - 1)\n",
            "Рекурсивно сравниваю крайние символы. Проверены строка из одного символа и чётная длина.",
            {"chars": "clean", "check": "same", "lo": "a", "hi": "b"},
        ),
    ],
}

# Special single-member families.
TEMPLATE_HEAVY = (
    "lab-01-fizzbuzz",
    "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n"
    "    result.extend(str(i) for i in range(1, n + 1))\n    return result\n",
    "Заготовка из шаблона, правила Fizz/Buzz ещё не реализованы.",
)
SYNTAX_ERROR = (
    "lab-02-word-count",
    "def word_count(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n"
    "    for word in text.lower().split()\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n",
    "Считаю слова после split; пунктуацию пока не обрабатываю.",
)

# Members of a family by layout: the derived variants made from its original.
FAMILY_LAYOUTS: list[tuple[str, ...]] = [
    ("exact_copy", "reformatted"),
    ("renamed", "modified"),
    ("reformatted", "renamed"),
    ("modified",),
]
RELATION_RANK = {"exact_copy": 0, "reformatted": 1, "renamed": 2, "modified": 3}
# Statements a "modified" copy inserts into the copied function (they change the token sequence).
SNIPPETS: list[list[str]] = [
    ["steps = 0", "steps += 1"],
    ["checked = []"],
    ["if False:", '    print("unreachable")'],
    ['debug = {"stage": "start"}'],
    ["for _ in range(0):", "    pass"],
    ["limit = 10 ** 6"],
]
_BLOCK_CONTINUATIONS = ("elif", "else", "except", "finally")
AI_TOOLS = ["ChatGPT", "Claude", "GitHub Copilot"]
COMMENTS = ["# main loop", "# edge case", "# build the answer", "# helper", "# done"]


# --- code transformations -----------------------------------------------------------------------------


def reformat(code: str, rng: random.Random) -> str:
    """Change layout only: 2-space indentation, extra comments and blank lines."""
    out: list[str] = ["# Solution", ""]
    for line in code.split("\n"):
        stripped = line.lstrip(" ")
        indent = (len(line) - len(stripped)) // 2
        new_line = " " * indent + stripped
        if stripped and not stripped.startswith(("def ", "if __name__")) and rng.random() < 0.3:
            new_line += "  " + rng.choice(COMMENTS)
        out.append(new_line)
        if stripped.startswith("return") and rng.random() < 0.5:
            out.append("")
    return "\n".join(out)


def rename(code: str, mapping: dict[str, str]) -> str:
    """Rename identifiers with ``tokenize`` so strings and attributes stay untouched."""
    if not mapping:
        return code
    lines = code.split("\n")
    edits: list[tuple[int, int, int, str]] = []
    previous = None
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        after_dot = previous is not None and previous.string == "."
        if tok.type == tokenize.NAME and tok.string in mapping and not after_dot:
            edits.append((tok.start[0], tok.start[1], tok.end[1], mapping[tok.string]))
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            previous = tok
    for row, start, end, new in sorted(edits, reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:start] + new + line[end:]
    return "\n".join(lines)


def _statement_rows(code: str) -> list[int]:
    """Rows where a logical line (statement) starts, found with ``tokenize``."""
    rows: list[int] = []
    at_start = True
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type in (tokenize.NEWLINE, tokenize.NL):
            at_start = tok.type == tokenize.NEWLINE or at_start
            continue
        if tok.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
            continue
        if at_start:
            rows.append(tok.start[0])
            at_start = False
    return rows


def modify(code: str, rng: random.Random) -> str:
    """Copy with a partial edit: 2-3 statements inserted inside the copied functions."""
    lines = code.split("\n")
    main_row = next(no for no, line in enumerate(lines, start=1) if line.startswith("if __name__"))
    candidates = [
        row
        for row in _statement_rows(code)
        if row < main_row
        and lines[row - 1].startswith(" ")
        and not lines[row - 1].lstrip().startswith(_BLOCK_CONTINUATIONS)
    ]
    rows = sorted(rng.sample(candidates, k=min(len(candidates), rng.randint(2, 3))), reverse=True)
    for row in rows:
        target = lines[row - 1]
        indent = target[: len(target) - len(target.lstrip())]
        lines[row - 1 : row - 1] = [indent + line for line in rng.choice(SNIPPETS)]
    result = "\n".join(lines)
    ast.parse(result)  # an edited copy must stay valid Python
    return result


def derive(code: str, relation: str, mapping: dict[str, str], rng: random.Random) -> str:
    if relation == "exact_copy":
        return code
    if relation == "reformatted":
        return reformat(code, rng)
    if relation == "renamed":
        return rename(code, mapping)
    if relation == "modified":
        return modify(code, rng)
    raise ValueError(relation)


# --- corpus ------------------------------------------------------------------------------------------


def template_lines(assignment_id: str) -> set[str]:
    template = next(a["allowed_template"] for a in ASSIGNMENTS if a["assignment_id"] == assignment_id)
    return {line.strip() for line in template.split("\n") if line.strip()}


def shares_only_template(code_a: str, code_b: str, assignment_id: str) -> bool:
    """True when every common non-empty line of two codes is a line of the allowed template."""
    common = {line.strip() for line in code_a.split("\n") if line.strip()} & {
        line.strip() for line in code_b.split("\n") if line.strip()
    }
    return bool(common) and common <= template_lines(assignment_id)


class CorpusBuilder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.records: list[dict[str, Any]] = []
        self.provenance: list[dict[str, Any]] = []
        self.family_assignment: dict[str, str] = {}
        self.family_layout: dict[str, int] = {}
        self._student = 0

    def next_student(self) -> str:
        self._student += 1
        return f"student-{self._student:03d}"

    def add(
        self,
        assignment_id: str,
        code: str,
        text: str,
        family_id: str,
        origin: str,
        derived_from: str | None,
        student_id: str | None = None,
        submitted_at: datetime | None = None,
    ) -> dict[str, Any]:
        used = self.rng.random() < 0.4
        tools = [self.rng.choice(AI_TOOLS)] if used else []
        record = {
            "submission_id": f"sub-{len(self.records) + 1:04d}",
            "assignment_id": assignment_id,
            "student_id": student_id or self.next_student(),
            "submitted_at": (
                submitted_at or BASE_TIME + timedelta(days=self.rng.randint(0, 6), minutes=self.rng.randint(0, 600))
            ).isoformat(),
            "text": text,
            "code": code,
            "ai_declaration": {
                "used": used,
                "tools": tools,
                "description": f"Использовал {tools[0]} для проверки граничных случаев." if used else "",
            },
        }
        self.records.append(record)
        self.provenance.append(
            {
                "submission_id": record["submission_id"],
                "family_id": family_id,
                "origin": origin,
                "derived_from": derived_from,
            }
        )
        self.family_assignment[family_id] = assignment_id
        return record

    def build(self) -> None:
        for assignment in ASSIGNMENTS:
            aid = assignment["assignment_id"]
            layouts = list(range(len(FAMILY_LAYOUTS)))
            self.rng.shuffle(layouts)
            for no, ((body, text, mapping), layout) in enumerate(zip(SOLUTIONS[aid], layouts, strict=True), start=1):
                family = f"{aid}/f{no}"
                code = body + _MAIN[aid]
                original = self.add(aid, code, text, family, "original", None)
                self.family_layout[family] = layout
                for relation in FAMILY_LAYOUTS[layout]:
                    self.add(
                        aid,
                        derive(code, relation, mapping, self.rng),
                        text,
                        family,
                        relation,
                        original["submission_id"],
                    )

        for no, (aid, body, text) in enumerate((TEMPLATE_HEAVY, SYNTAX_ERROR), start=5):
            self.add(aid, body + _MAIN[aid], text, f"{aid}/f{no}", "original", None)

        # Later versions by the same student: compared separately, never labeled as cross-student pairs.
        for aid in ("lab-03-transpose", "lab-04-palindrome"):
            first = next(
                r for r, p in zip(self.records, self.provenance, strict=True)
                if r["assignment_id"] == aid and p["origin"] == "original"
            )
            family = next(p["family_id"] for p in self.provenance if p["submission_id"] == first["submission_id"])
            self.add(
                aid,
                '"""Second version: input is validated."""\n' + first["code"],
                first["text"] + " Во второй версии добавлен docstring.",
                family,
                "resubmission",
                first["submission_id"],
                student_id=first["student_id"],
                submitted_at=datetime.fromisoformat(first["submitted_at"]) + timedelta(days=1),
            )

    def split(self) -> dict[str, str]:
        """Assign whole families to tune or eval.

        Stratified: every family layout lands on each side for half of the
        assignments, so both sides contain every kind of derivation. Special
        single-member families are split between the sides at random.
        """
        sides: dict[str, str] = {}
        for k, assignment in enumerate(ASSIGNMENTS):
            for family, aid in sorted(self.family_assignment.items()):
                if aid == assignment["assignment_id"] and family in self.family_layout:
                    sides[family] = "tune" if (self.family_layout[family] + k) % 2 == 0 else "eval"
        special = sorted(f for f in self.family_assignment if f not in self.family_layout)
        self.rng.shuffle(special)
        for i, family in enumerate(special):
            sides[family] = "tune" if i % 2 == 0 else "eval"
        return sides

    def pairs(self, sides: dict[str, str]) -> list[dict[str, Any]]:
        by_id = {r["submission_id"]: r for r in self.records}
        prov = {p["submission_id"]: p for p in self.provenance}
        result: list[dict[str, Any]] = []
        for a, b in combinations(self.records, 2):
            pa, pb = prov[a["submission_id"]], prov[b["submission_id"]]
            if a["assignment_id"] != b["assignment_id"] or a["student_id"] == b["student_id"]:
                continue
            if "resubmission" in (pa["origin"], pb["origin"]):
                continue
            side_a, side_b = sides[pa["family_id"]], sides[pb["family_id"]]
            if side_a != side_b:
                continue  # pairs across the split are not labeled: they would leak a family
            if pa["family_id"] == pb["family_id"]:
                origins = [o for o in (pa["origin"], pb["origin"]) if o != "original"]
                relation = max(origins, key=RELATION_RANK.__getitem__)
                template_only = False
            else:
                relation = "independent"
                template_only = shares_only_template(
                    by_id[a["submission_id"]]["code"], by_id[b["submission_id"]]["code"], a["assignment_id"]
                )
            result.append(
                {
                    "a": a["submission_id"],
                    "b": b["submission_id"],
                    "assignment_id": a["assignment_id"],
                    "family_a": pa["family_id"],
                    "family_b": pb["family_id"],
                    "derivation_relation": relation,
                    "template_only": template_only,
                    "split": side_a,
                }
            )
        return result


def check(records: list[dict[str, Any]], pairs: list[dict[str, Any]], sides: dict[str, str]) -> None:
    assert len(records) >= MIN_SUBMISSIONS, len(records)
    assert len(pairs) >= MIN_PAIRS, len(pairs)
    assert sum(p["template_only"] for p in pairs) >= MIN_TEMPLATE_PAIRS
    assert all(sides[p["family_a"]] == sides[p["family_b"]] == p["split"] for p in pairs)
    for side in ("tune", "eval"):
        relations = {p["derivation_relation"] for p in pairs if p["split"] == side}
        assert relations == {"exact_copy", "reformatted", "renamed", "modified", "independent"}, (side, relations)
        assert any(p["template_only"] for p in pairs if p["split"] == side), side


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("datasets/lab2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = CorpusBuilder(random.Random(args.seed))
    builder.build()
    sides = builder.split()
    pairs = builder.pairs(sides)
    check(builder.records, pairs, sides)

    family_of = {p["submission_id"]: p["family_id"] for p in builder.provenance}
    split = {
        "seed": args.seed,
        "families": dict(sorted(sides.items())),
        "submissions": {sid: sides[family_of[sid]] for sid in sorted(family_of)},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "assignments.json").write_text(
        json.dumps({"assignments": ASSIGNMENTS}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.out / "submissions.jsonl", builder.records)
    write_jsonl(args.out / "provenance.jsonl", builder.provenance)
    write_jsonl(args.out / "pairs.jsonl", pairs)
    (args.out / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    relations: dict[str, int] = {}
    for p in pairs:
        relations[p["derivation_relation"]] = relations.get(p["derivation_relation"], 0) + 1
    print(
        f"corpus written to {args.out} (seed={args.seed}): {len(builder.records)} submissions, "
        f"{len(sides)} families, {len(pairs)} labeled pairs {relations}, "
        f"template_only={sum(p['template_only'] for p in pairs)}"
    )


if __name__ == "__main__":
    main()
