"""Line handling shared by analyzers.

Line numbers in evidence always refer to the original submission content,
counted the way an editor counts them: 1-based, split on ``\\n`` only.
"""

from __future__ import annotations


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
