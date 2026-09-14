"""Canonical reanchor naming for structural attention analysis."""

from .lookback import InformationDiagnostics, LookbackAnalyzer, LookbackResult

ReanchorAnalyzer = LookbackAnalyzer
ReanchorResult = LookbackResult

__all__ = ["InformationDiagnostics", "ReanchorAnalyzer", "ReanchorResult"]