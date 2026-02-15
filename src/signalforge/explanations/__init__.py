"""SHAP-based explanation system for SignalForge predictions."""

from .narrative import NarrativeGenerator
from .schemas import ExplanationConfig, FeatureContribution, PredictionExplanation
from .shap_explainer import SHAPExplainer
from .visualization import ExplanationVisualizer

__all__ = [
    "ExplanationConfig",
    "ExplanationVisualizer",
    "FeatureContribution",
    "NarrativeGenerator",
    "PredictionExplanation",
    "SHAPExplainer",
]
