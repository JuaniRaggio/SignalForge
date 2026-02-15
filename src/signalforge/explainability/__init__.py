"""Explainability service layer for the Dashboard Explanation Service.

This module provides database-integrated explanation services for model predictions,
including SHAP-based explanations, feature importance tracking, trader-friendly
translation of technical SHAP values, and dashboard widgets for visualization.
"""

from .explanation_service import ExplanationService
from .explanation_translator import ExplanationTranslator
from .shap_calculator import SHAPCalculator
from .widgets import (
    AccuracyByFeatureWidget,
    ConfidenceIndicatorWidget,
    ExplainabilityDashboardService,
    FeatureImportanceWidget,
    ModelComparisonWidget,
    PredictionExplainerWidget,
)

__all__ = [
    "ExplanationService",
    "ExplanationTranslator",
    "SHAPCalculator",
    "AccuracyByFeatureWidget",
    "ConfidenceIndicatorWidget",
    "ExplainabilityDashboardService",
    "FeatureImportanceWidget",
    "ModelComparisonWidget",
    "PredictionExplainerWidget",
]
