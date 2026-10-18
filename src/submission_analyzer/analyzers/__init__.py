"""Built-in analyzers and the analyzer contract."""

from submission_analyzer.analyzers.base import AnalysisContext, Analyzer, SubmissionReader, build_report
from submission_analyzer.analyzers.length import LengthAnalyzer
from submission_analyzer.analyzers.model_stub import ModelStubAnalyzer
from submission_analyzer.analyzers.requirements import RequirementsAnalyzer
from submission_analyzer.analyzers.template import TemplateAnalyzer

__all__ = [
    "AnalysisContext",
    "Analyzer",
    "LengthAnalyzer",
    "ModelStubAnalyzer",
    "RequirementsAnalyzer",
    "SubmissionReader",
    "TemplateAnalyzer",
    "build_report",
]
