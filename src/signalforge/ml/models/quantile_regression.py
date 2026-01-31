"""Quantile Regression models for prediction interval generation.

This module implements quantile regression models that produce prediction intervals
rather than point forecasts. Quantile regression estimates conditional quantiles
of the target variable, enabling uncertainty quantification and risk assessment.

Key features:
- Multiple quantile predictions (e.g., 0.1, 0.5, 0.9 for 80% prediction intervals)
- Calibration metrics to assess prediction interval quality
- Linear and gradient boosting implementations
- Integration with technical indicators and feature engineering

The module provides:
- QuantileRegressor: Linear quantile regression using sklearn
- QuantileGradientBoostingRegressor: Non-linear gradient boosting approach
- QuantilePrediction: Structured prediction output with intervals
- Calibration and scoring utilities

All models extend BasePredictor and integrate with MLflow for experiment tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor as SKLearnQuantileRegressor

from signalforge.core.logging import get_logger
from signalforge.ml.models.base import BasePredictor
from signalforge.ml.training.mlflow_config import log_metrics, log_params

logger = get_logger(__name__)


@dataclass
class QuantileRegressionConfig:
    """Configuration for quantile regression models.

    Attributes:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
                  Must be in range (0, 1). Median (0.5) recommended.
        alpha: Regularization strength for linear models (L1 penalty).
              Higher values increase regularization. Default: 0.1
        max_iter: Maximum iterations for solver convergence. Default: 1000
        features: Feature column names to use. If None, auto-selects
                 OHLCV and common technical indicators.
        solver: Solver algorithm for linear models. Options:
               'highs-ds': Dual simplex (fast, recommended)
               'highs-ipm': Interior point method (robust)
               'highs': Automatic selection
        n_lags: Number of lag features to create for time series. Default: 5
    """

    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    alpha: float = 0.1
    max_iter: int = 1000
    features: list[str] | None = None
    solver: Literal["highs-ds", "highs-ipm", "highs"] = "highs-ds"
    n_lags: int = 5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.quantiles:
            raise ValueError("At least one quantile must be specified")

        for q in self.quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantile {q} must be in range (0, 1)")

        if len(self.quantiles) != len(set(self.quantiles)):
            raise ValueError("Quantiles must be unique")

        if self.alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {self.alpha}")

        if self.max_iter < 1:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")

        if self.n_lags < 0:
            raise ValueError(f"n_lags must be non-negative, got {self.n_lags}")


@dataclass
class QuantilePrediction:
    """Structured output for quantile predictions.

    Attributes:
        point_forecast: Point estimate (median/0.5 quantile if available).
        lower_bound: Lower prediction bound (smallest quantile).
        upper_bound: Upper prediction bound (largest quantile).
        all_quantiles: Mapping of quantile levels to predicted values.
        coverage: Expected coverage probability (upper_quantile - lower_quantile).
                 E.g., 0.8 for 0.1-0.9 interval.
        timestamp: Optional timestamp for the prediction.
    """

    point_forecast: float
    lower_bound: float
    upper_bound: float
    all_quantiles: dict[float, float]
    coverage: float
    timestamp: Any | None = None

    def __post_init__(self) -> None:
        """Validate prediction consistency."""
        if self.lower_bound > self.point_forecast:
            logger.warning(
                f"Lower bound {self.lower_bound} exceeds point forecast {self.point_forecast}"
            )
        if self.point_forecast > self.upper_bound:
            logger.warning(
                f"Point forecast {self.point_forecast} exceeds upper bound {self.upper_bound}"
            )
        if not 0 <= self.coverage <= 1:
            raise ValueError(f"Coverage must be in [0, 1], got {self.coverage}")


class QuantileRegressor(BasePredictor):
    """Linear Quantile Regression for prediction intervals.

    This model fits separate linear quantile regressors for each specified
    quantile level. It uses L1 regularization and is robust to outliers.

    Quantile regression minimizes the tilted absolute loss:
        rho(u) = u * (quantile - I(u < 0))
    where u = y - y_pred.

    Attributes:
        config: Configuration object with model hyperparameters.
        _models: Dictionary mapping quantile to fitted sklearn model.
        _training_data: Cached training data for prediction continuation.
        _target_column: Name of target column.
        _feature_columns: List of feature column names used.
        _fitted: Training status flag.

    Examples:
        Basic usage with default quantiles:

        >>> import polars as pl
        >>> from signalforge.ml.models.quantile_regression import (
        ...     QuantileRegressor,
        ...     QuantileRegressionConfig
        ... )
        >>>
        >>> config = QuantileRegressionConfig(
        ...     quantiles=[0.05, 0.5, 0.95],
        ...     alpha=0.01
        ... )
        >>> model = QuantileRegressor(config)
        >>> model.fit(df, target_column="close")
        >>> predictions = model.predict(horizon=10)
    """

    def __init__(self, config: QuantileRegressionConfig) -> None:
        """Initialize quantile regressor.

        Args:
            config: Model configuration with quantiles and hyperparameters.
        """
        self.config = config
        self._models: dict[float, SKLearnQuantileRegressor] = {}
        self._training_data: pl.DataFrame | None = None
        self._target_column: str = "close"
        self._feature_columns: list[str] = []
        self._fitted: bool = False

    def _select_features(self, df: pl.DataFrame) -> list[str]:
        """Auto-select features if not specified in config.

        Prioritizes OHLCV data and common technical indicators.

        Args:
            df: Input DataFrame.

        Returns:
            List of feature column names.

        Raises:
            ValueError: If no suitable features found.
        """
        if self.config.features is not None:
            missing = set(self.config.features) - set(df.columns)
            if missing:
                raise ValueError(f"Features not found in DataFrame: {missing}")
            return self.config.features

        # Auto-select features: OHLCV + technical indicators
        ohlcv = ["open", "high", "low", "close", "volume"]
        technical = [
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "obv",
        ]

        available_features = []
        for col in ohlcv + technical:
            if col in df.columns and col != self._target_column:
                available_features.append(col)

        if not available_features:
            raise ValueError(
                "No suitable features found. Provide features in config or "
                "ensure OHLCV/technical indicator columns exist."
            )

        logger.info(f"Auto-selected {len(available_features)} features: {available_features}")
        return available_features

    def _create_lag_features(self, df: pl.DataFrame, target_col: str, n_lags: int) -> pl.DataFrame:
        """Create lagged features for time series prediction.

        Args:
            df: Input DataFrame.
            target_col: Target column name.
            n_lags: Number of lags to create.

        Returns:
            DataFrame with lag features added.
        """
        if n_lags == 0:
            return df

        result = df.clone()
        for lag in range(1, n_lags + 1):
            lag_col = f"{target_col}_lag_{lag}"
            result = result.with_columns(pl.col(target_col).shift(lag).alias(lag_col))
            if lag_col not in self._feature_columns:
                self._feature_columns.append(lag_col)

        return result

    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train quantile regression models on historical data.

        Fits separate model for each quantile in config. Creates lag features
        and handles missing values automatically.

        Args:
            df: Training data with features and target column.
            target_column: Name of target variable. Default: "close".

        Raises:
            ValueError: If DataFrame empty, missing columns, or insufficient data.
            RuntimeError: If model fitting fails.
        """
        if df.is_empty():
            raise ValueError("Cannot fit model on empty DataFrame")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        self._target_column = target_column
        self._feature_columns = self._select_features(df)

        logger.info(
            f"Fitting quantile regression with {len(self._feature_columns)} features "
            f"for quantiles {self.config.quantiles}"
        )

        # Create lag features
        df_with_lags = self._create_lag_features(df, target_column, self.config.n_lags)

        # Remove rows with null values (from lags or missing data)
        df_clean = df_with_lags.drop_nulls()

        if df_clean.height < 10:
            raise ValueError(
                f"Insufficient training data after cleaning: {df_clean.height} rows. "
                f"Need at least 10 rows."
            )

        # Extract features and target
        X = df_clean.select(self._feature_columns).to_numpy()
        y = df_clean.select(target_column).to_numpy().ravel()

        # Log parameters to MLflow (if active run exists)
        try:
            log_params(
                {
                    "model_type": "QuantileRegressor",
                    "quantiles": self.config.quantiles,
                    "alpha": self.config.alpha,
                    "max_iter": self.config.max_iter,
                    "n_features": len(self._feature_columns),
                    "n_lags": self.config.n_lags,
                    "solver": self.config.solver,
                    "training_samples": len(y),
                }
            )
        except RuntimeError:
            # No active MLflow run, skip logging
            logger.debug("Skipping MLflow parameter logging (no active run)")
            pass

        # Fit model for each quantile
        for quantile in self.config.quantiles:
            logger.info(f"Fitting model for quantile {quantile}")
            model = SKLearnQuantileRegressor(
                quantile=quantile,
                alpha=self.config.alpha,
                solver=self.config.solver,
                solver_options={"max_iter": self.config.max_iter},
            )

            try:
                model.fit(X, y)
                self._models[quantile] = model
                logger.info(f"Successfully fitted quantile {quantile}")
            except Exception as e:
                logger.error(f"Failed to fit quantile {quantile}: {e}")
                raise RuntimeError(f"Model fitting failed for quantile {quantile}: {e}") from e

        self._training_data = df_with_lags
        self._fitted = True
        logger.info("Quantile regression training completed successfully")

    def predict(self, horizon: int) -> pl.DataFrame:
        """Generate predictions with prediction intervals.

        Creates multi-step forecasts with quantile-based uncertainty bounds.
        Uses recursive prediction where each step's forecast feeds into next.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with columns:
            - timestamp: Forecast timestamps
            - prediction: Point forecast (median if available, else mean)
            - lower_bound: Lower prediction interval
            - upper_bound: Upper prediction interval
            - coverage: Expected coverage probability
            - quantile_<q>: Prediction for each quantile q

        Raises:
            RuntimeError: If model not fitted.
            ValueError: If horizon not positive.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if horizon < 1:
            raise ValueError(f"Horizon must be positive, got {horizon}")

        if self._training_data is None:
            raise RuntimeError("Training data not found")

        logger.info(f"Generating predictions for horizon={horizon}")

        # Get last known values for recursive prediction
        last_row = self._training_data.tail(1)
        last_timestamp = last_row.select("timestamp").to_series()[0]

        predictions_list: list[dict[str, Any]] = []

        # Use last complete feature set as starting point
        current_features = (
            self._training_data.drop_nulls().tail(1).select(self._feature_columns).to_numpy()[0]
        )

        for step in range(1, horizon + 1):
            # Predict for all quantiles
            step_quantiles: dict[float, float] = {}
            for quantile in sorted(self.config.quantiles):
                pred = self._models[quantile].predict(current_features.reshape(1, -1))[0]
                step_quantiles[quantile] = float(pred)

            # Create QuantilePrediction object
            sorted_quantiles = sorted(step_quantiles.keys())
            lower_bound = step_quantiles[sorted_quantiles[0]]
            upper_bound = step_quantiles[sorted_quantiles[-1]]

            # Use median as point forecast if available, else mean
            if 0.5 in step_quantiles:
                point_forecast = step_quantiles[0.5]
            else:
                point_forecast = float(np.mean(list(step_quantiles.values())))

            coverage = sorted_quantiles[-1] - sorted_quantiles[0]

            # Calculate timestamp
            pred_timestamp = last_timestamp + timedelta(days=step)

            pred_dict: dict[str, Any] = {
                "timestamp": pred_timestamp,
                "prediction": point_forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "coverage": coverage,
            }

            # Add individual quantile predictions
            for q, val in step_quantiles.items():
                pred_dict[f"quantile_{q}"] = val

            predictions_list.append(pred_dict)

            # Update features for next step (simple approach: shift and update target lags)
            # In production, you'd update all lag features properly
            if self.config.n_lags > 0:
                # Shift lag features and add new prediction
                for lag in range(self.config.n_lags, 1, -1):
                    lag_idx = self._feature_columns.index(f"{self._target_column}_lag_{lag}")
                    prev_lag_idx = self._feature_columns.index(
                        f"{self._target_column}_lag_{lag - 1}"
                    )
                    current_features[lag_idx] = current_features[prev_lag_idx]

                # Set lag_1 to current prediction
                if f"{self._target_column}_lag_1" in self._feature_columns:
                    lag_1_idx = self._feature_columns.index(f"{self._target_column}_lag_1")
                    current_features[lag_1_idx] = point_forecast

        result_df = pl.DataFrame(predictions_list)
        logger.info(f"Generated {len(predictions_list)} predictions")
        return result_df

    def evaluate(self, test_df: pl.DataFrame) -> dict[str, float]:
        """Evaluate model on test data with interval-based metrics.

        Computes both point forecast metrics and interval quality metrics
        including coverage and Winkler score.

        Args:
            test_df: Test data with actual values.

        Returns:
            Dictionary with metrics:
            - rmse: Root mean squared error (point forecast)
            - mae: Mean absolute error (point forecast)
            - mape: Mean absolute percentage error (point forecast)
            - empirical_coverage: Actual coverage rate
            - expected_coverage: Theoretical coverage
            - coverage_deviation: Abs difference between empirical and expected
            - winkler_score: Interval scoring metric (lower is better)
            - interval_width: Average prediction interval width

        Raises:
            RuntimeError: If model not fitted.
            ValueError: If test_df empty or missing columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        if test_df.is_empty():
            raise ValueError("Test DataFrame cannot be empty")

        if self._target_column not in test_df.columns:
            raise ValueError(f"Target column '{self._target_column}' not found in test data")

        logger.info("Evaluating quantile regression model")

        # Create lag features for test data
        test_with_lags = self._create_lag_features(test_df, self._target_column, self.config.n_lags)
        test_clean = test_with_lags.drop_nulls()

        if test_clean.height == 0:
            raise ValueError("No valid test samples after creating lag features")

        # Get features and actual values
        X_test = test_clean.select(self._feature_columns).to_numpy()
        y_actual = test_clean.select(self._target_column).to_numpy().ravel()

        # Generate predictions for all quantiles
        predictions: dict[float, np.ndarray] = {}
        for quantile in self.config.quantiles:
            predictions[quantile] = self._models[quantile].predict(X_test)

        # Calculate point forecast metrics (using median or mean)
        if 0.5 in predictions:
            y_pred = predictions[0.5]
        else:
            y_pred = np.mean(np.array(list(predictions.values())), axis=0)

        # Point forecast metrics
        errors = y_pred - y_actual
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        mape = float(np.mean(np.abs(errors / y_actual)) * 100)

        # Interval metrics
        sorted_quantiles = sorted(self.config.quantiles)
        lower_preds = predictions[sorted_quantiles[0]]
        upper_preds = predictions[sorted_quantiles[-1]]
        expected_coverage = sorted_quantiles[-1] - sorted_quantiles[0]

        # Empirical coverage: fraction of actuals within intervals
        in_interval = (y_actual >= lower_preds) & (y_actual <= upper_preds)
        empirical_coverage = float(np.mean(in_interval))
        coverage_deviation = abs(empirical_coverage - expected_coverage)

        # Winkler score: penalizes wide intervals and violations
        winkler = winkler_score(lower_preds, upper_preds, y_actual, 1 - expected_coverage)

        # Average interval width
        interval_width = float(np.mean(upper_preds - lower_preds))

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "empirical_coverage": empirical_coverage,
            "expected_coverage": expected_coverage,
            "coverage_deviation": coverage_deviation,
            "winkler_score": winkler,
            "interval_width": interval_width,
        }

        # Log metrics to MLflow (if active run exists)
        try:
            log_metrics(metrics)
        except RuntimeError:
            # No active MLflow run, skip logging
            logger.debug("Skipping MLflow metrics logging (no active run)")
            pass

        logger.info(
            f"Evaluation complete - RMSE: {rmse:.4f}, Coverage: "
            f"{empirical_coverage:.2%} (expected {expected_coverage:.2%})"
        )

        return metrics

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained.

        Returns:
            True if model is fitted and ready for prediction.
        """
        return self._fitted


class QuantileGradientBoostingRegressor(BasePredictor):
    """Gradient Boosting Quantile Regression for non-linear relationships.

    Uses sklearn's GradientBoostingRegressor with quantile loss function.
    Better suited for complex, non-linear patterns compared to linear
    quantile regression.

    Gradient boosting builds an ensemble of weak learners (decision trees)
    sequentially, each correcting errors of the previous ensemble.

    Attributes:
        config: Configuration with quantiles and hyperparameters.
        n_estimators: Number of boosting stages. Default: 100
        learning_rate: Step size shrinkage to prevent overfitting. Default: 0.1
        max_depth: Maximum tree depth. Default: 3
        subsample: Fraction of samples for fitting base learners. Default: 1.0
        _models: Dictionary of fitted models per quantile.
        _training_data: Cached training data.
        _target_column: Target variable name.
        _feature_columns: Feature names used.
        _fitted: Training status.

    Examples:
        >>> from signalforge.ml.models.quantile_regression import (
        ...     QuantileGradientBoostingRegressor,
        ...     QuantileRegressionConfig
        ... )
        >>>
        >>> config = QuantileRegressionConfig(quantiles=[0.1, 0.5, 0.9])
        >>> model = QuantileGradientBoostingRegressor(
        ...     config,
        ...     n_estimators=200,
        ...     learning_rate=0.05,
        ...     max_depth=5
        ... )
        >>> model.fit(df, target_column="close")
        >>> predictions = model.predict(horizon=20)
    """

    def __init__(
        self,
        config: QuantileRegressionConfig,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        """Initialize gradient boosting quantile regressor.

        Args:
            config: Model configuration with quantiles.
            n_estimators: Number of boosting stages. More trees can improve
                         accuracy but increase computation time.
            learning_rate: Learning rate shrinks contribution of each tree.
                          Lower values require more estimators.
            max_depth: Maximum depth of trees. Higher allows more complex
                      patterns but risks overfitting.
            subsample: Fraction of samples used for fitting. Values < 1.0
                      enable stochastic gradient boosting.
            random_state: Random seed for reproducibility.
        """
        self.config = config
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

        self._models: dict[float, GradientBoostingRegressor] = {}
        self._training_data: pl.DataFrame | None = None
        self._target_column: str = "close"
        self._feature_columns: list[str] = []
        self._fitted: bool = False

    def _select_features(self, df: pl.DataFrame) -> list[str]:
        """Auto-select features if not specified. Same as QuantileRegressor."""
        if self.config.features is not None:
            missing = set(self.config.features) - set(df.columns)
            if missing:
                raise ValueError(f"Features not found in DataFrame: {missing}")
            return self.config.features

        ohlcv = ["open", "high", "low", "close", "volume"]
        technical = [
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "obv",
        ]

        available_features = []
        for col in ohlcv + technical:
            if col in df.columns and col != self._target_column:
                available_features.append(col)

        if not available_features:
            raise ValueError(
                "No suitable features found. Provide features in config or "
                "ensure OHLCV/technical indicator columns exist."
            )

        logger.info(f"Auto-selected {len(available_features)} features: {available_features}")
        return available_features

    def _create_lag_features(self, df: pl.DataFrame, target_col: str, n_lags: int) -> pl.DataFrame:
        """Create lagged features for time series prediction."""
        if n_lags == 0:
            return df

        result = df.clone()
        for lag in range(1, n_lags + 1):
            lag_col = f"{target_col}_lag_{lag}"
            result = result.with_columns(pl.col(target_col).shift(lag).alias(lag_col))
            if lag_col not in self._feature_columns:
                self._feature_columns.append(lag_col)

        return result

    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train gradient boosting models for each quantile.

        Args:
            df: Training data with features and target.
            target_column: Name of target variable. Default: "close".

        Raises:
            ValueError: If data invalid or insufficient.
            RuntimeError: If fitting fails.
        """
        if df.is_empty():
            raise ValueError("Cannot fit model on empty DataFrame")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        self._target_column = target_column
        self._feature_columns = self._select_features(df)

        logger.info(
            f"Fitting gradient boosting quantile regression with "
            f"{len(self._feature_columns)} features for quantiles {self.config.quantiles}"
        )

        # Create lag features
        df_with_lags = self._create_lag_features(df, target_column, self.config.n_lags)
        df_clean = df_with_lags.drop_nulls()

        if df_clean.height < 20:
            raise ValueError(
                f"Insufficient training data: {df_clean.height} rows. Need at least 20."
            )

        X = df_clean.select(self._feature_columns).to_numpy()
        y = df_clean.select(target_column).to_numpy().ravel()

        # Log parameters to MLflow (if active run exists)
        try:
            log_params(
                {
                    "model_type": "QuantileGradientBoosting",
                    "quantiles": self.config.quantiles,
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "subsample": self.subsample,
                    "n_features": len(self._feature_columns),
                    "n_lags": self.config.n_lags,
                    "training_samples": len(y),
                }
            )
        except RuntimeError:
            # No active MLflow run, skip logging
            logger.debug("Skipping MLflow parameter logging (no active run)")
            pass

        # Fit model for each quantile
        for quantile in self.config.quantiles:
            logger.info(f"Fitting gradient boosting for quantile {quantile}")

            # Convert quantile to alpha for loss='quantile'
            alpha = quantile

            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                random_state=self.random_state,
                verbose=0,
            )

            try:
                model.fit(X, y)
                self._models[quantile] = model
                logger.info(f"Successfully fitted quantile {quantile}")
            except Exception as e:
                logger.error(f"Failed to fit quantile {quantile}: {e}")
                raise RuntimeError(f"Model fitting failed for quantile {quantile}: {e}") from e

        self._training_data = df_with_lags
        self._fitted = True
        logger.info("Gradient boosting quantile regression training completed")

    def predict(self, horizon: int) -> pl.DataFrame:
        """Generate predictions with prediction intervals.

        Same interface as QuantileRegressor.predict().

        Args:
            horizon: Number of periods to forecast.

        Returns:
            DataFrame with prediction intervals and quantiles.

        Raises:
            RuntimeError: If model not fitted.
            ValueError: If horizon invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if horizon < 1:
            raise ValueError(f"Horizon must be positive, got {horizon}")

        if self._training_data is None:
            raise RuntimeError("Training data not found")

        logger.info(f"Generating predictions for horizon={horizon}")

        last_row = self._training_data.tail(1)
        last_timestamp = last_row.select("timestamp").to_series()[0]

        predictions_list: list[dict[str, Any]] = []
        current_features = (
            self._training_data.drop_nulls().tail(1).select(self._feature_columns).to_numpy()[0]
        )

        for step in range(1, horizon + 1):
            step_quantiles: dict[float, float] = {}
            for quantile in sorted(self.config.quantiles):
                pred = self._models[quantile].predict(current_features.reshape(1, -1))[0]
                step_quantiles[quantile] = float(pred)

            sorted_quantiles = sorted(step_quantiles.keys())
            lower_bound = step_quantiles[sorted_quantiles[0]]
            upper_bound = step_quantiles[sorted_quantiles[-1]]

            if 0.5 in step_quantiles:
                point_forecast = step_quantiles[0.5]
            else:
                point_forecast = float(np.mean(list(step_quantiles.values())))

            coverage = sorted_quantiles[-1] - sorted_quantiles[0]
            pred_timestamp = last_timestamp + timedelta(days=step)

            pred_dict: dict[str, Any] = {
                "timestamp": pred_timestamp,
                "prediction": point_forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "coverage": coverage,
            }

            for q, val in step_quantiles.items():
                pred_dict[f"quantile_{q}"] = val

            predictions_list.append(pred_dict)

            # Update lag features
            if self.config.n_lags > 0:
                for lag in range(self.config.n_lags, 1, -1):
                    lag_idx = self._feature_columns.index(f"{self._target_column}_lag_{lag}")
                    prev_lag_idx = self._feature_columns.index(
                        f"{self._target_column}_lag_{lag - 1}"
                    )
                    current_features[lag_idx] = current_features[prev_lag_idx]

                if f"{self._target_column}_lag_1" in self._feature_columns:
                    lag_1_idx = self._feature_columns.index(f"{self._target_column}_lag_1")
                    current_features[lag_1_idx] = point_forecast

        result_df = pl.DataFrame(predictions_list)
        logger.info(f"Generated {len(predictions_list)} predictions")
        return result_df

    def evaluate(self, test_df: pl.DataFrame) -> dict[str, float]:
        """Evaluate model with interval metrics. Same as QuantileRegressor."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        if test_df.is_empty():
            raise ValueError("Test DataFrame cannot be empty")

        if self._target_column not in test_df.columns:
            raise ValueError(f"Target column '{self._target_column}' not found")

        logger.info("Evaluating gradient boosting quantile regression")

        test_with_lags = self._create_lag_features(test_df, self._target_column, self.config.n_lags)
        test_clean = test_with_lags.drop_nulls()

        if test_clean.height == 0:
            raise ValueError("No valid test samples")

        X_test = test_clean.select(self._feature_columns).to_numpy()
        y_actual = test_clean.select(self._target_column).to_numpy().ravel()

        predictions: dict[float, np.ndarray] = {}
        for quantile in self.config.quantiles:
            predictions[quantile] = self._models[quantile].predict(X_test)

        if 0.5 in predictions:
            y_pred = predictions[0.5]
        else:
            y_pred = np.mean(np.array(list(predictions.values())), axis=0)

        errors = y_pred - y_actual
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        mape = float(np.mean(np.abs(errors / y_actual)) * 100)

        sorted_quantiles = sorted(self.config.quantiles)
        lower_preds = predictions[sorted_quantiles[0]]
        upper_preds = predictions[sorted_quantiles[-1]]
        expected_coverage = sorted_quantiles[-1] - sorted_quantiles[0]

        in_interval = (y_actual >= lower_preds) & (y_actual <= upper_preds)
        empirical_coverage = float(np.mean(in_interval))
        coverage_deviation = abs(empirical_coverage - expected_coverage)

        winkler = winkler_score(lower_preds, upper_preds, y_actual, 1 - expected_coverage)
        interval_width = float(np.mean(upper_preds - lower_preds))

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "empirical_coverage": empirical_coverage,
            "expected_coverage": expected_coverage,
            "coverage_deviation": coverage_deviation,
            "winkler_score": winkler,
            "interval_width": interval_width,
        }

        # Log metrics to MLflow (if active run exists)
        try:
            log_metrics(metrics)
        except RuntimeError:
            # No active MLflow run, skip logging
            logger.debug("Skipping MLflow metrics logging (no active run)")
            pass

        logger.info(
            f"Evaluation complete - RMSE: {rmse:.4f}, Coverage: "
            f"{empirical_coverage:.2%} (expected {expected_coverage:.2%})"
        )

        return metrics

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._fitted


def create_quantile_regressor(
    config: QuantileRegressionConfig, method: Literal["linear", "gbm"] = "linear"
) -> BasePredictor:
    """Factory function to create quantile regression models.

    Args:
        config: Configuration with quantiles and hyperparameters.
        method: Model type - "linear" for linear quantile regression,
               "gbm" for gradient boosting.

    Returns:
        Instantiated quantile regression model.

    Raises:
        ValueError: If method is invalid.

    Examples:
        >>> config = QuantileRegressionConfig(quantiles=[0.1, 0.5, 0.9])
        >>> model = create_quantile_regressor(config, method="gbm")
    """
    if method == "linear":
        return QuantileRegressor(config)
    elif method == "gbm":
        return QuantileGradientBoostingRegressor(config)
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'linear' or 'gbm'")


def calculate_coverage(
    predictions: pl.DataFrame,
    actuals: pl.Series,
    lower_col: str = "lower_bound",
    upper_col: str = "upper_bound",
) -> float:
    """Calculate empirical coverage of prediction intervals.

    Measures the fraction of actual values that fall within predicted intervals.
    Well-calibrated intervals should have empirical coverage close to expected.

    Args:
        predictions: DataFrame with lower and upper bounds.
        actuals: Actual observed values.
        lower_col: Name of lower bound column. Default: "lower_bound"
        upper_col: Name of upper bound column. Default: "upper_bound"

    Returns:
        Coverage rate in [0, 1]. E.g., 0.85 means 85% of actuals in intervals.

    Raises:
        ValueError: If columns missing or length mismatch.

    Examples:
        >>> import polars as pl
        >>> preds = pl.DataFrame({
        ...     "lower_bound": [95.0, 98.0, 101.0],
        ...     "upper_bound": [105.0, 108.0, 111.0]
        ... })
        >>> actuals = pl.Series([100.0, 107.0, 115.0])
        >>> coverage = calculate_coverage(preds, actuals)
        >>> print(f"Coverage: {coverage:.2%}")
    """
    if lower_col not in predictions.columns or upper_col not in predictions.columns:
        raise ValueError(f"Columns '{lower_col}' and '{upper_col}' must exist in predictions")

    if len(predictions) != len(actuals):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, actuals={len(actuals)}")

    lower = predictions.select(lower_col).to_series()
    upper = predictions.select(upper_col).to_series()

    in_interval = (actuals >= lower) & (actuals <= upper)
    coverage = float(in_interval.sum() / len(actuals))

    return coverage


def winkler_score(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray, alpha: float) -> float:
    """Calculate Winkler score for prediction interval quality.

    The Winkler score penalizes both wide intervals and coverage violations.
    Lower scores indicate better interval predictions.

    Score = interval_width + (2/alpha) * penalty for violations
    where penalty = amount by which actual exceeds bounds.

    Args:
        lower: Array of lower bounds.
        upper: Array of upper bounds.
        actual: Array of actual values.
        alpha: Significance level (1 - coverage). E.g., 0.2 for 80% interval.

    Returns:
        Average Winkler score. Lower is better.

    Raises:
        ValueError: If arrays have different lengths or alpha invalid.

    Examples:
        >>> import numpy as np
        >>> lower = np.array([95.0, 98.0])
        >>> upper = np.array([105.0, 108.0])
        >>> actual = np.array([100.0, 110.0])  # Second exceeds upper
        >>> score = winkler_score(lower, upper, actual, alpha=0.2)
    """
    if not (len(lower) == len(upper) == len(actual)):
        raise ValueError("All arrays must have same length")

    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    interval_width = upper - lower

    # Penalty for violations
    lower_violation = np.maximum(0, lower - actual)
    upper_violation = np.maximum(0, actual - upper)
    penalty = (2 / alpha) * (lower_violation + upper_violation)

    scores = interval_width + penalty
    return float(np.mean(scores))
