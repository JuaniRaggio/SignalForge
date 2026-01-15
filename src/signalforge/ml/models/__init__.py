"""Machine learning models for time series prediction.

This module provides various prediction models for financial time series,
including baseline statistical models and deep learning models.

All models inherit from the BasePredictor abstract class and provide
consistent interfaces for training, prediction, and evaluation.

Examples:
    Basic usage of ARIMA predictor:

    >>> import polars as pl
    >>> from signalforge.ml.models.baseline import ARIMAPredictor
    >>>
    >>> df = pl.DataFrame({
    ...     "timestamp": pl.date_range(start="2024-01-01", periods=100, interval="1d"),
    ...     "close": [100.0 + i * 0.5 for i in range(100)],
    ... })
    >>> model = ARIMAPredictor(order=(1, 1, 1))
    >>> model.fit(df, target_column="close")
    >>> predictions = model.predict(horizon=10)
"""

from signalforge.ml.models.base import BasePredictor
from signalforge.ml.models.baseline import ARIMAPredictor, RollingMeanPredictor

__all__ = [
    "BasePredictor",
    "ARIMAPredictor",
    "RollingMeanPredictor",
]
