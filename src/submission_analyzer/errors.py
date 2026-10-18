"""Domain exceptions.

Every rejected record is explained by one or more :class:`ErrorDetail` objects
(``code``, ``field``, ``message``). The exception class tells *which kind* of
problem occurred:

* :class:`FormatError` — the line is not a JSON object at all;
* :class:`SchemaError` — valid JSON that violates the submission schema;
* :class:`ConflictError` — a valid submission whose id is already taken by
  different content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorKind(StrEnum):
    FORMAT = "format"
    SCHEMA = "schema"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: str
    field: str | None
    message: str

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "field": self.field, "message": self.message}


class IngestionError(Exception):
    """Base class for all reasons a record can be rejected."""

    kind: ErrorKind

    def __init__(self, details: list[ErrorDetail] | tuple[ErrorDetail, ...]) -> None:
        if not details:
            raise ValueError("IngestionError requires at least one ErrorDetail")
        self.details: tuple[ErrorDetail, ...] = tuple(details)
        super().__init__("; ".join(f"{d.code} [{d.field}]: {d.message}" for d in self.details))


class FormatError(IngestionError):
    kind = ErrorKind.FORMAT


class SchemaError(IngestionError):
    kind = ErrorKind.SCHEMA


class ConflictError(IngestionError):
    kind = ErrorKind.CONFLICT


class ConfigError(Exception):
    """Invalid processing configuration."""


class DataSourceError(Exception):
    """An auxiliary data file (e.g. assignments.json) cannot be loaded."""
