"""Model inference and explainability for SignalForge.

This module provides inference capabilities and model explainability
using SHAP (SHapley Additive exPlanations) values. It enables:

- Single and batch prediction explanations
- Feature importance analysis
- Human-readable summaries with financial context
- Visualization data generation for plots

Key Components:
    ModelExplainer: SHAP-based explainer for ML models
    ExplanationResult: Complete explanation for a prediction
    FeatureImportance: Individual feature contribution
    ExplainerConfig: Configuration for explainer behavior

Examples:
    Basic usage with a trained model:

    >>> from signalforge.ml.inference import ModelExplainer, ExplainerConfig
    >>> import polars as pl
    >>>
    >>> # Configure explainer
    >>> config = ExplainerConfig(method="tree", max_features=10)
    >>> explainer = ModelExplainer(config)
    >>>
    >>> # Explain a prediction
    >>> X = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
    >>> result = explainer.explain(model, X)
    >>> print(result.summary_text)

    Batch processing:

    >>> # Explain multiple predictions
    >>> X_batch = pl.DataFrame({
    ...     "rsi_14": [65.0, 45.0, 55.0],
    ...     "macd": [0.5, -0.3, 0.1]
    ... })
    >>> results = explainer.explain_batch(model, X_batch)
    >>>
    >>> # Get global feature importance
    >>> importances = explainer.get_feature_importance(model, X_batch)
    >>> for imp in importances[:5]:
    ...     print(f"{imp.feature}: {imp.importance:.4f} ({imp.direction})")
"""

from signalforge.ml.inference.explainer import (
    BaseExplainer,
    ExplainerConfig,
    ExplanationResult,
    FeatureImportance,
    ModelExplainer,
    generate_explanation_text,
    plot_summary,
    plot_waterfall,
)

__all__ = [
    "BaseExplainer",
    "ModelExplainer",
    "ExplainerConfig",
    "ExplanationResult",
    "FeatureImportance",
    "generate_explanation_text",
    "plot_waterfall",
    "plot_summary",
]
