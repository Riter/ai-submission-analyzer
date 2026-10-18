"""Submission sources: where raw JSONL lines come from.

A source yields ``(line_no, raw_line)`` pairs and is used as a context
manager, so whoever opens it also closes it — on normal completion, on an
exception and on early exit (``break``) alike.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import IO, Protocol, Self


class SubmissionSource(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def __iter__(self) -> Iterator[tuple[int, str]]: ...


class JsonlFileSource:
    """Reads a JSONL file lazily, line by line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[str] | None = None

    @property
    def closed(self) -> bool:
        return self._fh is None or self._fh.closed

    def __enter__(self) -> Self:
        if self._fh is not None:
            raise RuntimeError(f"{self.path} is already open")
        self._fh = self.path.open(encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
        # Returning None lets any exception propagate to the caller.

    def __iter__(self) -> Iterator[tuple[int, str]]:
        if self._fh is None or self._fh.closed:
            raise RuntimeError("JsonlFileSource must be used inside a 'with' block")
        for line_no, line in enumerate(self._fh, start=1):
            yield line_no, line.rstrip("\r\n")


class InMemorySource:
    """Same contract over lines already in memory (tests, programmatic use)."""

    def __init__(self, lines: Iterable[str]) -> None:
        self._lines = tuple(lines)
        self._open = False

    @property
    def closed(self) -> bool:
        return not self._open

    def __enter__(self) -> Self:
        self._open = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._open = False

    def __iter__(self) -> Iterator[tuple[int, str]]:
        if not self._open:
            raise RuntimeError("InMemorySource must be used inside a 'with' block")
        for line_no, line in enumerate(self._lines, start=1):
            yield line_no, line
