"""Training corpus metadata: provenance of examples, labeled pairs and the tune/eval split.

Provenance and pair labels describe how a *synthetic example was made*. They
are not a verdict about plagiarism and are never passed to analyzers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class Origin(StrEnum):
    ORIGINAL = "original"
    EXACT_COPY = "exact_copy"
    REFORMATTED = "reformatted"
    RENAMED = "renamed"
    RESUBMISSION = "resubmission"  # a later version by the same student


class DerivationRelation(StrEnum):
    EXACT_COPY = "exact_copy"
    REFORMATTED = "reformatted"
    RENAMED = "renamed"
    INDEPENDENT = "independent"


class SplitName(StrEnum):
    TUNE = "tune"
    EVAL = "eval"


@dataclass(frozen=True, slots=True)
class Provenance:
    submission_id: str
    family_id: str
    origin: Origin
    derived_from: str | None


@dataclass(frozen=True, slots=True)
class LabeledPair:
    a: str
    b: str
    assignment_id: str
    family_a: str
    family_b: str
    derivation_relation: DerivationRelation
    template_only: bool
    split: SplitName

    @property
    def is_derived(self) -> bool:
        return self.derivation_relation is not DerivationRelation.INDEPENDENT


@dataclass(frozen=True, slots=True)
class Split:
    """Each family belongs to exactly one side; submissions follow their family."""

    families: Mapping[str, SplitName]
    submissions: Mapping[str, SplitName]

    def submission_ids(self, side: SplitName) -> frozenset[str]:
        return frozenset(sid for sid, s in self.submissions.items() if s is side)


def check_no_family_leakage(
    split: Split,
    pairs: Iterable[LabeledPair],
    provenance: Iterable[Provenance] = (),
) -> list[str]:
    """Return every place where one family's examples end up on both sides of the split."""
    problems: list[str] = []
    for item in provenance:
        family_side = split.families.get(item.family_id)
        submission_side = split.submissions.get(item.submission_id)
        if family_side is None:
            problems.append(f"family {item.family_id} is not assigned to a split")
        elif submission_side is not family_side:
            problems.append(
                f"{item.submission_id} is in {submission_side} but its family {item.family_id} is in {family_side}"
            )
    for pair in pairs:
        sides = {split.families.get(pair.family_a), split.families.get(pair.family_b), pair.split}
        if len(sides) != 1:
            problems.append(f"pair {pair.a}/{pair.b} mixes splits: {sorted(map(str, sides))}")
    return problems
