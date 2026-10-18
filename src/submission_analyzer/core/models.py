"""Domain models.

Models are frozen: once a submission is accepted its original content never changes.
Only raw student content (``text``, ``code``) lives here; derived features are a
separate concern keyed by ``content_hash`` and are added in later labs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AIDeclaration:
    """Student's self-declaration of AI tool usage.

    This is a declaration, not a ground-truth label: it must never be used as a
    training label for text provenance.
    """

    used: bool
    tools: tuple[str, ...]
    description: str


@dataclass(frozen=True, slots=True)
class Submission:
    """One immutable version of a student's lab submission."""

    submission_id: str
    assignment_id: str
    student_id: str
    submitted_at: datetime
    text: str
    code: str
    ai_declaration: AIDeclaration
    content_hash: str


@dataclass(frozen=True, slots=True)
class Assignment:
    """Assignment requirements that submissions are checked against."""

    assignment_id: str
    title: str
    requirements: tuple[str, ...]
    allowed_template: str
