"""Line handling shared by analyzers.

Line numbers in evidence always refer to the original submission content,
counted the way an editor counts them: 1-based, split on ``\\n`` only.
"""

from __future__ import annotations

from collections.abc import Iterable


def source_lines(content: str) -> list[str]:
    """Split content into lines as an editor shows them (no trailing phantom line)."""
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


def count_nonempty_lines(content: str) -> int:
    return sum(1 for line in source_lines(content) if line.strip())


def line_runs(line_numbers: Iterable[int]) -> list[tuple[int, int]]:
    """Group line numbers into consecutive ranges: [1, 2, 3, 7] -> [(1, 3), (7, 7)]."""
    runs: list[tuple[int, int]] = []
    for no in sorted(set(line_numbers)):
        if runs and runs[-1][1] == no - 1:
            runs[-1] = (runs[-1][0], no)
        else:
            runs.append((no, no))
    return runs
