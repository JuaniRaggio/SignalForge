"""Training module for SignalForge ML models.

This module provides:
- MLflow experiment tracking configuration
- Model training helpers
- Experiment management utilities
- Walk-forward validation for time series
- Complete training pipeline with feature engineering
"""

from signalforge.ml.training.mlflow_config import (
    log_metrics,
    log_model,
    log_params,
    setup_experiment,
    start_run,
)
from signalforge.ml.training.mlflow_utils import MLflowTracker
from signalforge.ml.training.pipeline import TrainingPipeline, TrainingResult
from signalforge.ml.training.validation import ValidationResult, WalkForwardValidator

__all__ = [
    "setup_experiment",
    "log_params",
    "log_metrics",
    "log_model",
    "start_run",
    "WalkForwardValidator",
    "ValidationResult",
    "MLflowTracker",
    "TrainingPipeline",
    "TrainingResult",
]
