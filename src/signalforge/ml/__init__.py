"""Machine Learning module for SignalForge.

This module provides ML capabilities including:
- Feature engineering for technical indicators
- Model training and evaluation
- Prediction pipelines
- Model explainability with SHAP
- Walk-forward validation for time series
- MLflow experiment tracking
- Production serving with batch prediction and caching
- Model optimization with ONNX and quantization
- A/B testing and canary deployments
"""

from signalforge.ml.models.base import BasePredictor, PredictionResult
from signalforge.ml.optimization.onnx_converter import ONNXConverter
from signalforge.ml.optimization.quantization import ModelQuantizer, QuantizationConfig
from signalforge.ml.serving.batch_predictor import BatchPredictor
from signalforge.ml.serving.model_router import ModelRouter
from signalforge.ml.serving.prediction_cache import PredictionCache
from signalforge.ml.training.mlflow_utils import MLflowTracker
from signalforge.ml.training.validation import ValidationResult, WalkForwardValidator

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "WalkForwardValidator",
    "ValidationResult",
    "MLflowTracker",
    # Serving
    "BatchPredictor",
    "PredictionCache",
    "ModelRouter",
    # Optimization
    "ONNXConverter",
    "ModelQuantizer",
    "QuantizationConfig",
]
