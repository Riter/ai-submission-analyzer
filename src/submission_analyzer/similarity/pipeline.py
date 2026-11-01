"""Lazy pipeline ``read → validate → features → index``.

Every stage is a generator: a line is read, validated and turned into features
before the next line is read. Only the final index holds data for the whole
corpus (see :mod:`submission_analyzer.similarity.index` on its memory).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

from submission_analyzer.core.config import SimilarityConfig
from submission_analyzer.core.models import Assignment, Submission
from submission_analyzer.ingestion.service import ImportReport, IngestionService
from submission_analyzer.similarity.features import CodeFeatures, extract_features
from submission_analyzer.similarity.index import FeatureIndex


def iter_accepted(
    service: IngestionService,
    lines: Iterable[tuple[int, str]],
    report: ImportReport,
) -> Iterator[Submission]:
    """Validate lines one by one, yield newly accepted submissions.

    Rejected lines and duplicates are recorded in ``report``; they never stop
    the stream.
    """
    for result, submission in service.iter_numbered(lines):
        report.lines.append(result)
        if submission is not None:
            yield submission


def iter_features(
    submissions: Iterable[Submission],
    assignments: Mapping[str, Assignment],
    config: SimilarityConfig,
) -> Iterator[CodeFeatures]:
    for submission in submissions:
        assignment = assignments.get(submission.assignment_id)
        template = assignment.allowed_template if assignment is not None else ""
        yield extract_features(submission, template, config)


def build_index(features: Iterable[CodeFeatures]) -> FeatureIndex:
    return FeatureIndex(features)
