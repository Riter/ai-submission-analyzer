"""Generate the synthetic lab 1 dataset deterministically.

Usage:
    python scripts/generate_lab1_data.py [--out datasets/lab1] [--seed 42]

Produces:
    assignments.json           3 assignments with an allowed code template
    submissions.jsonl          valid submissions + 1 duplicate + 1 conflict + 1 new version
    invalid_submissions.jsonl  one line per error scenario (+ one valid line at the end)

All data is synthetic; no real student work is used. Re-running with the same
seed produces byte-identical files.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

MSK = timezone(timedelta(hours=3))
BASE_TIME = datetime(2026, 9, 14, 10, 0, tzinfo=MSK)

ASSIGNMENTS: list[dict[str, Any]] = [
    {
        "assignment_id": "lab-01-fizzbuzz",
        "title": "FizzBuzz",
        "requirements": [
            "Function fizzbuzz(n) returns a list of strings for 1..n",
            "Multiples of 3 are 'Fizz', of 5 are 'Buzz', of both are 'FizzBuzz'",
            "Report explains the approach",
        ],
        "allowed_template": "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n    return result\n",
    },
    {
        "assignment_id": "lab-02-word-count",
        "title": "Word frequency",
        "requirements": [
            "Function word_count(text) returns a dict word -> count",
            "Words are compared case-insensitively",
            "Report describes how punctuation is handled",
        ],
        "allowed_template": "def word_count(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n    return counts\n",
    },
    {
        "assignment_id": "lab-03-transpose",
        "title": "Matrix transpose",
        "requirements": [
            "Function transpose(matrix) returns the transposed matrix",
            "Empty matrix returns an empty list",
            "Report states the time complexity",
        ],
        "allowed_template": "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    return []\n",
    },
]

# Solution bodies per assignment: (code, report text). Several stylistic variants
# so that different students do not produce identical code.
SOLUTIONS: dict[str, list[tuple[str, str]]] = {
    "lab-01-fizzbuzz": [
        (
            "def fizzbuzz(n: int) -> list[str]:\n    result: list[str] = []\n"
            "    for i in range(1, n + 1):\n        if i % 15 == 0:\n            result.append(\"FizzBuzz\")\n"
            "        elif i % 3 == 0:\n            result.append(\"Fizz\")\n        elif i % 5 == 0:\n"
            "            result.append(\"Buzz\")\n        else:\n            result.append(str(i))\n    return result\n",
            "Перебираю числа от 1 до n и проверяю делимость сначала на 15, затем на 3 и 5.",
        ),
        (
            "def fizzbuzz(n: int) -> list[str]:\n    out = []\n    for k in range(1, n + 1):\n"
            "        word = (\"Fizz\" if k % 3 == 0 else \"\") + (\"Buzz\" if k % 5 == 0 else \"\")\n"
            "        out.append(word or str(k))\n    return out\n",
            "Строю слово конкатенацией: Fizz для кратных 3 и Buzz для кратных 5; пустое слово заменяю числом.",
        ),
        (
            "def fizzbuzz(n: int) -> list[str]:\n"
            "    return [\"FizzBuzz\" if i % 15 == 0 else \"Fizz\" if i % 3 == 0 else \"Buzz\" if i % 5 == 0 else str(i)\n"
            "            for i in range(1, n + 1)]\n",
            "Решение одним списковым включением с цепочкой условных выражений.",
        ),
    ],
    "lab-02-word-count": [
        (
            "import re\n\n\ndef word_count(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n"
            "    for word in re.findall(r\"[\\w']+\", text.lower()):\n        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n",
            "Привожу текст к нижнему регистру и выделяю слова регулярным выражением; пунктуация отбрасывается.",
        ),
        (
            "from collections import Counter\nimport string\n\n\ndef word_count(text: str) -> dict[str, int]:\n"
            "    cleaned = text.lower().translate(str.maketrans(\"\", \"\", string.punctuation))\n"
            "    return dict(Counter(cleaned.split()))\n",
            "Удаляю знаки пунктуации через str.translate и считаю слова с помощью Counter.",
        ),
    ],
    "lab-03-transpose": [
        (
            "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    if not matrix:\n        return []\n"
            "    return [list(row) for row in zip(*matrix)]\n",
            "Использую zip(*matrix). Сложность O(n*m), где n и m — размеры матрицы.",
        ),
        (
            "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    rows = len(matrix)\n"
            "    cols = len(matrix[0]) if rows else 0\n"
            "    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]\n",
            "Двойной цикл по индексам. Время O(rows*cols), дополнительная память того же порядка.",
        ),
    ],
}

AI_TOOLS = ["ChatGPT", "Claude", "GitHub Copilot"]


def make_submission(rng: random.Random, sub_no: int, student_no: int, assignment_id: str) -> dict[str, Any]:
    code, text = rng.choice(SOLUTIONS[assignment_id])
    used = rng.random() < 0.4
    tools = [rng.choice(AI_TOOLS)] if used else []
    description = f"Использовал {tools[0]} для проверки граничных случаев." if used else ""
    submitted_at = BASE_TIME + timedelta(days=rng.randint(0, 6), minutes=rng.randint(0, 600))
    return {
        "submission_id": f"sub-{sub_no:04d}",
        "assignment_id": assignment_id,
        "student_id": f"student-{student_no:03d}",
        "submitted_at": submitted_at.isoformat(),
        "text": text,
        "code": code,
        "ai_declaration": {"used": used, "tools": tools, "description": description},
    }


def build_valid(rng: random.Random) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sub_no = 1
    for assignment in ASSIGNMENTS:
        for student_no in range(1, 5):
            if assignment["assignment_id"] == "lab-03-transpose" and student_no == 4:
                continue
            records.append(make_submission(rng, sub_no, student_no, assignment["assignment_id"]))
            sub_no += 1
    return records


def build_submissions(rng: random.Random) -> list[dict[str, Any]]:
    records = build_valid(rng)

    duplicate = json.loads(json.dumps(records[0]))  # exact resubmission

    conflict = json.loads(json.dumps(records[1]))  # same id, different content
    conflict["code"] += "\n# edited after submission\n"

    new_version = json.loads(json.dumps(records[1]))  # proper new version: new id
    new_version["submission_id"] = f"sub-{len(records) + 1:04d}"
    new_version["code"] = conflict["code"]
    new_version["submitted_at"] = (datetime.fromisoformat(records[1]["submitted_at"]) + timedelta(hours=2)).isoformat()

    return [*records, duplicate, conflict, new_version]


def build_invalid_lines(rng: random.Random) -> list[str]:
    def base(sub_id: str) -> dict[str, Any]:
        record = make_submission(rng, 0, 9, "lab-01-fizzbuzz")
        record["submission_id"] = sub_id
        return record

    lines: list[str] = []

    lines.append('{"submission_id": "bad-0001", "assignment_id": "lab-01-fizzbuzz", "student_id": ')  # invalid_json
    lines.append("[1, 2, 3]")  # not_an_object

    r = base("bad-0003")
    r["submitted_at"] = "2026-13-45T25:00:00+03:00"  # invalid_timestamp
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0004")
    r["submitted_at"] = "2026-09-15T12:00:00"  # naive_timestamp
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0005")
    r["student_id"] = "   "  # empty_value
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0006")
    r["grade"] = 5  # unknown_field
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0007")
    r["student_id"] = 42  # invalid_type
    r["ai_declaration"]["used"] = "yes"  # invalid_type
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0008")
    del r["code"]  # missing_field
    del r["ai_declaration"]  # missing_field
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0009")
    r["ai_declaration"] = {"used": False, "tools": ["ChatGPT"], "description": ""}  # invalid_ai_declaration
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0010")
    r["assignment_id"] = "lab-99-unknown"  # unknown_assignment
    lines.append(json.dumps(r, ensure_ascii=False))

    r = base("bad-0011")
    r["text"] = ""
    r["code"] = "  \n"  # empty_value (no content)
    lines.append(json.dumps(r, ensure_ascii=False))

    lines.append(json.dumps(base("ok-0012"), ensure_ascii=False))  # valid: import continues after errors
    return lines


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("datasets/lab1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    (args.out / "assignments.json").write_text(
        json.dumps({"assignments": ASSIGNMENTS}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.out / "submissions.jsonl", [json.dumps(r, ensure_ascii=False) for r in build_submissions(rng)])
    write_jsonl(args.out / "invalid_submissions.jsonl", build_invalid_lines(rng))
    print(f"dataset written to {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
