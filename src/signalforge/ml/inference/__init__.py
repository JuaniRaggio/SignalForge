"""Model inference and explainability for SignalForge.

This module provides comprehensive inference capabilities including:

- Production prediction service with caching
- ONNX Runtime integration for optimized inference
- Model registry for versioning and management
- SHAP-based explainability

Key Components:
    PredictionService: Main service for generating predictions
    PredictionResponse: Structured prediction result
    ONNXPredictor: ONNX Runtime wrapper for fast inference
    ModelRegistry: Registry for managing trained models
    ModelInfo: Model metadata and versioning
    ModelExplainer: SHAP-based explainer for ML models
    ExplanationResult: Complete explanation for a prediction

Examples:
    Production prediction service:

    >>> from signalforge.ml.inference import PredictionService, ModelRegistry
    >>> from signalforge.ml.features import TechnicalFeatureEngine
    >>>
    >>> registry = ModelRegistry("models/")
    >>> features = TechnicalFeatureEngine()
    >>> service = PredictionService(registry, features)
    >>>
    >>> result = await service.predict("AAPL", horizon_days=5)
    >>> print(f"Predicted return: {result.predicted_return:.2%}")

    ONNX Runtime inference:

    >>> from signalforge.ml.inference import ONNXPredictor
    >>> import polars as pl
    >>>
    >>> predictor = ONNXPredictor("models/lstm_v1.onnx")
    >>> X = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
    >>> predictions = predictor.predict(X)

    Model registry:

    >>> from signalforge.ml.inference import ModelRegistry
    >>>
    >>> registry = ModelRegistry("models/")
    >>> model_id = registry.register(
    ...     model,
    ...     metrics={"mse": 0.025},
    ...     tags={"type": "lstm"}
    ... )
    >>> registry.set_default(model_id)
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
from signalforge.ml.inference.model_registry import ModelInfo, ModelRegistry
from signalforge.ml.inference.onnx_runtime import ONNXPredictor
from signalforge.ml.inference.predictor import PredictionResponse, PredictionService

__all__ = [
    # Prediction service
    "PredictionService",
    "PredictionResponse",
    # ONNX Runtime
    "ONNXPredictor",
    # Model registry
    "ModelRegistry",
    "ModelInfo",
    # Explainability
    "BaseExplainer",
    "ModelExplainer",
    "ExplainerConfig",
    "ExplanationResult",
    "FeatureImportance",
    "generate_explanation_text",
    "plot_waterfall",
    "plot_summary",
]
