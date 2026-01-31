"""Machine learning models for time series prediction.

This module provides various prediction models for financial time series,
including baseline statistical models, deep learning models, and ensemble methods.

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

    Ensemble usage:

    >>> from signalforge.ml.models.ensemble import create_ensemble, EnsembleConfig
    >>>
    >>> config = EnsembleConfig(
    ...     models=["arima", "rolling_mean"],
    ...     method="weighted",
    ...     optimize_weights=True
    ... )
    >>> ensemble = create_ensemble(config)
    >>> ensemble.add_model("arima", ARIMAPredictor(order=(1, 1, 1)))
    >>> ensemble.add_model("rolling_mean", RollingMeanPredictor(window=20))
    >>> ensemble.fit(df, target_column="close")
    >>> result = ensemble.predict(horizon=10)
"""

from signalforge.ml.models.base import BasePredictor
from signalforge.ml.models.baseline import ARIMAPredictor, RollingMeanPredictor
from signalforge.ml.models.ensemble import (
    BaseEnsemble,
    EnsembleConfig,
    EnsemblePrediction,
    StackingEnsemble,
    WeightedEnsemble,
    create_ensemble,
    optimize_weights,
)
from signalforge.ml.models.quantile_regression import (
    QuantileGradientBoostingRegressor,
    QuantilePrediction,
    QuantileRegressionConfig,
    QuantileRegressor,
    calculate_coverage,
    create_quantile_regressor,
    winkler_score,
)

__all__ = [
    "BasePredictor",
    "ARIMAPredictor",
    "RollingMeanPredictor",
    "BaseEnsemble",
    "WeightedEnsemble",
    "StackingEnsemble",
    "EnsembleConfig",
    "EnsemblePrediction",
    "create_ensemble",
    "optimize_weights",
    "QuantileRegressor",
    "QuantileGradientBoostingRegressor",
    "QuantileRegressionConfig",
    "QuantilePrediction",
    "create_quantile_regressor",
    "calculate_coverage",
    "winkler_score",
]
