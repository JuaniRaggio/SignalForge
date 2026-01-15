"""Baseline statistical models for time series prediction.

This module implements baseline models that serve as benchmarks for more
complex machine learning and deep learning models. These models provide:
- Simple, interpretable predictions
- Fast training and inference
- Reliable performance baselines for comparison

The baseline models include:
- ARIMAPredictor: Classical statistical model for time series
- RollingMeanPredictor: Simple naive forecast based on moving average

All models integrate with MLflow for experiment tracking and follow the
BasePredictor interface.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl
from statsmodels.tsa.arima.model import ARIMA

from signalforge.core.logging import get_logger
from signalforge.ml.models.base import BasePredictor
from signalforge.ml.training.mlflow_config import log_metrics, log_params

logger = get_logger(__name__)


def calculate_metrics(actual: pl.Series, predicted: pl.Series) -> dict[str, float]:
    """Calculate evaluation metrics for predictions.

    This function computes standard regression and forecasting metrics
    to evaluate model performance.

    Args:
        actual: Series of actual values
        predicted: Series of predicted values

    Returns:
        Dictionary containing:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - mape: Mean Absolute Percentage Error
        - direction_accuracy: Percentage of correct directional predictions

    Raises:
        ValueError: If series lengths don't match or contain invalid values

    Examples:
        >>> import polars as pl
        >>> actual = pl.Series([100.0, 101.0, 102.0, 103.0])
        >>> predicted = pl.Series([100.5, 100.8, 102.2, 103.1])
        >>> metrics = calculate_metrics(actual, predicted)
        >>> print(metrics["rmse"])
    """
    if len(actual) != len(predicted):
        raise ValueError(
            f"Series length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )

    if len(actual) == 0:
        raise ValueError("Cannot calculate metrics for empty series")

    # Remove any null values
    mask = actual.is_not_null() & predicted.is_not_null()
    actual_clean = actual.filter(mask)
    predicted_clean = predicted.filter(mask)

    if len(actual_clean) == 0:
        raise ValueError("No valid (non-null) values to calculate metrics")

    # Calculate errors
    errors = predicted_clean - actual_clean
    squared_errors = errors * errors
    absolute_errors = errors.abs()

    # RMSE: Root Mean Squared Error
    rmse = float(squared_errors.mean() ** 0.5)

    # MAE: Mean Absolute Error
    mae = float(absolute_errors.mean())

    # MAPE: Mean Absolute Percentage Error
    # Avoid division by zero by filtering out values close to zero
    non_zero_mask = actual_clean.abs() > 1e-10
    if non_zero_mask.sum() > 0:
        percentage_errors = (absolute_errors.filter(non_zero_mask) / actual_clean.filter(non_zero_mask).abs()) * 100
        mape = float(percentage_errors.mean())
    else:
        mape = float("inf")

    # Direction Accuracy: Percentage of correct directional predictions
    # Calculate changes in actual and predicted values
    if len(actual_clean) > 1:
        actual_direction = (actual_clean.slice(1) - actual_clean.slice(0, len(actual_clean) - 1)) > 0
        predicted_direction = (predicted_clean.slice(1) - predicted_clean.slice(0, len(predicted_clean) - 1)) > 0
        direction_correct = actual_direction == predicted_direction
        direction_accuracy = float(direction_correct.sum() / len(direction_correct) * 100)
    else:
        direction_accuracy = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "direction_accuracy": direction_accuracy,
    }


class ARIMAPredictor(BasePredictor):
    """ARIMA model for time series prediction.

    AutoRegressive Integrated Moving Average (ARIMA) is a classical statistical
    model for time series forecasting. It combines:
    - AR (AutoRegressive): Uses past values to predict future
    - I (Integrated): Differencing to make series stationary
    - MA (Moving Average): Uses past forecast errors

    This implementation wraps statsmodels ARIMA with MLflow integration.

    Attributes:
        order: ARIMA order (p, d, q) where:
               p = number of lag observations
               d = degree of differencing
               q = size of moving average window
        seasonal_order: Optional seasonal ARIMA order (P, D, Q, s)
        _model: Fitted ARIMA model (None until fit() is called)
        _fitted_model: Result object from model fitting
        _training_data: Last training DataFrame for prediction continuation
        _target_column: Name of the target column
        _fitted: Boolean flag indicating if model is trained

    Examples:
        Basic usage:

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

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Initialize ARIMA predictor.

        Args:
            order: ARIMA order (p, d, q). Defaults to (1, 1, 1).
            seasonal_order: Optional seasonal order (P, D, Q, s). If None, no seasonality.

        Raises:
            ValueError: If order parameters are negative.
        """
        if len(order) != 3:
            raise ValueError("order must be a tuple of 3 integers (p, d, q)")
        if any(x < 0 for x in order):
            raise ValueError("order parameters must be non-negative")
        if seasonal_order is not None:
            if len(seasonal_order) != 4:
                raise ValueError("seasonal_order must be a tuple of 4 integers (P, D, Q, s)")
            if any(x < 0 for x in seasonal_order):
                raise ValueError("seasonal_order parameters must be non-negative")

        self.order = order
        self.seasonal_order = seasonal_order
        self._model: ARIMA | None = None
        self._fitted_model: Any = None
        self._training_data: pl.DataFrame | None = None
        self._target_column: str = "close"
        self._fitted: bool = False

        logger.debug(
            "arima_predictor_initialized",
            order=order,
            seasonal_order=seasonal_order,
        )

    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train the ARIMA model on historical data.

        Args:
            df: Input DataFrame with time series data.
                Must contain at least timestamp and target columns.
            target_column: Name of the column to predict. Defaults to "close".

        Raises:
            ValueError: If DataFrame is empty or missing required columns.
            RuntimeError: If model fitting fails.
        """
        if df.height == 0:
            raise ValueError("Cannot fit model on empty DataFrame")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        # Sort by timestamp to ensure chronological order
        df_sorted = df.sort("timestamp")

        # Extract target series
        target_series = df_sorted[target_column].to_numpy()

        logger.info(
            "fitting_arima_model",
            n_samples=len(target_series),
            order=self.order,
            seasonal_order=self.seasonal_order,
            target_column=target_column,
        )

        try:
            # Log parameters to MLflow if in active run
            try:
                log_params(
                    {
                        "model_type": "ARIMA",
                        "order_p": self.order[0],
                        "order_d": self.order[1],
                        "order_q": self.order[2],
                        "seasonal": self.seasonal_order is not None,
                        "target_column": target_column,
                        "n_samples": len(target_series),
                    }
                )
            except RuntimeError:
                # No active MLflow run, skip logging
                logger.debug("no_active_mlflow_run", action="skipping parameter logging")

            # Fit ARIMA model
            self._model = ARIMA(
                target_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
            )
            self._fitted_model = self._model.fit()

            # Store training data for prediction continuation
            self._training_data = df_sorted
            self._target_column = target_column
            self._fitted = True

            logger.info(
                "arima_model_fitted",
                aic=self._fitted_model.aic,
                bic=self._fitted_model.bic,
            )

        except Exception as e:
            logger.error(
                "arima_fitting_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to fit ARIMA model: {e}")

    def predict(self, horizon: int) -> pl.DataFrame:
        """Generate predictions for future periods.

        Args:
            horizon: Number of periods to predict into the future.

        Returns:
            DataFrame with columns:
            - timestamp: Future timestamps
            - prediction: Predicted values
            - lower_ci: Lower confidence interval (95%)
            - upper_ci: Upper confidence interval (95%)

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If horizon is not positive.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        logger.info("generating_arima_predictions", horizon=horizon)

        try:
            # Generate predictions with confidence intervals
            forecast = self._fitted_model.forecast(steps=horizon)
            forecast_result = self._fitted_model.get_forecast(steps=horizon)
            forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

            # Get last timestamp from training data
            assert self._training_data is not None
            last_timestamp = self._training_data["timestamp"].max()

            # Generate future timestamps
            # Infer frequency from training data
            if self._training_data.height >= 2:
                time_diff = (
                    self._training_data["timestamp"][1] - self._training_data["timestamp"][0]
                )
            else:
                # Default to 1 day if only one sample
                time_diff = timedelta(days=1)

            # Create future timestamps
            future_timestamps = [
                last_timestamp + time_diff * (i + 1) for i in range(horizon)
            ]

            # Create prediction DataFrame
            predictions_df = pl.DataFrame(
                {
                    "timestamp": future_timestamps,
                    "prediction": forecast,
                    "lower_ci": forecast_ci[:, 0],
                    "upper_ci": forecast_ci[:, 1],
                }
            )

            logger.debug(
                "arima_predictions_generated",
                horizon=horizon,
                first_prediction=float(forecast[0]),
            )

            return predictions_df

        except Exception as e:
            logger.error(
                "arima_prediction_failed",
                error=str(e),
                horizon=horizon,
                exc_info=True,
            )
            raise RuntimeError(f"Failed to generate predictions: {e}")

    def evaluate(self, test_df: pl.DataFrame) -> dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            test_df: DataFrame with actual values for comparison.
                     Must contain timestamp and target column.

        Returns:
            Dictionary of evaluation metrics (RMSE, MAE, MAPE, direction_accuracy).

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If test_df is empty or missing required columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation. Call fit() first.")

        if test_df.height == 0:
            raise ValueError("Cannot evaluate on empty DataFrame")

        if self._target_column not in test_df.columns:
            raise ValueError(
                f"Target column '{self._target_column}' not found in test DataFrame"
            )

        logger.info("evaluating_arima_model", test_samples=test_df.height)

        try:
            # Sort test data by timestamp
            test_sorted = test_df.sort("timestamp")

            # Generate predictions for test period
            horizon = test_sorted.height
            predictions_df = self.predict(horizon=horizon)

            # Extract actual and predicted values
            actual = test_sorted[self._target_column]
            predicted = predictions_df["prediction"]

            # Calculate metrics
            metrics = calculate_metrics(actual, predicted)

            logger.info(
                "arima_evaluation_complete",
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                mape=metrics["mape"],
                direction_accuracy=metrics["direction_accuracy"],
            )

            # Log metrics to MLflow if in active run
            try:
                log_metrics(metrics)
            except RuntimeError:
                # No active MLflow run, skip logging
                logger.debug("no_active_mlflow_run", action="skipping metrics logging")

            return metrics

        except Exception as e:
            logger.error(
                "arima_evaluation_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to evaluate model: {e}")

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained.

        Returns:
            True if model is fitted and ready for prediction.
        """
        return self._fitted and self._fitted_model is not None


class RollingMeanPredictor(BasePredictor):
    """Naive baseline predictor using rolling mean.

    This simple predictor uses the rolling mean of recent values to forecast
    future values. It serves as a basic baseline to compare against more
    sophisticated models.

    The prediction for each future period is the mean of the last 'window'
    observed values. This model assumes no trend or seasonality.

    Attributes:
        window: Number of recent periods to average for prediction
        _training_data: Last training DataFrame for prediction
        _target_column: Name of the target column
        _fitted: Boolean flag indicating if model is trained

    Examples:
        Basic usage:

        >>> import polars as pl
        >>> from signalforge.ml.models.baseline import RollingMeanPredictor
        >>>
        >>> df = pl.DataFrame({
        ...     "timestamp": pl.date_range(start="2024-01-01", periods=100, interval="1d"),
        ...     "close": [100.0 + i * 0.5 for i in range(100)],
        ... })
        >>> model = RollingMeanPredictor(window=10)
        >>> model.fit(df, target_column="close")
        >>> predictions = model.predict(horizon=5)
    """

    def __init__(self, window: int = 20) -> None:
        """Initialize rolling mean predictor.

        Args:
            window: Number of recent periods to average. Defaults to 20.

        Raises:
            ValueError: If window is not positive.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")

        self.window = window
        self._training_data: pl.DataFrame | None = None
        self._target_column: str = "close"
        self._fitted: bool = False

        logger.debug("rolling_mean_predictor_initialized", window=window)

    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train the rolling mean predictor.

        For this simple model, training just stores the data and validates it.

        Args:
            df: Input DataFrame with time series data.
            target_column: Name of the column to predict. Defaults to "close".

        Raises:
            ValueError: If DataFrame is empty or missing required columns,
                       or if insufficient data for the window size.
        """
        if df.height == 0:
            raise ValueError("Cannot fit model on empty DataFrame")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        if df.height < self.window:
            raise ValueError(
                f"Insufficient data: need at least {self.window} samples, got {df.height}"
            )

        # Sort by timestamp
        df_sorted = df.sort("timestamp")

        logger.info(
            "fitting_rolling_mean_model",
            n_samples=df_sorted.height,
            window=self.window,
            target_column=target_column,
        )

        try:
            # Log parameters to MLflow if in active run
            try:
                log_params(
                    {
                        "model_type": "RollingMean",
                        "window": self.window,
                        "target_column": target_column,
                        "n_samples": df_sorted.height,
                    }
                )
            except RuntimeError:
                # No active MLflow run, skip logging
                logger.debug("no_active_mlflow_run", action="skipping parameter logging")

            # Store training data
            self._training_data = df_sorted
            self._target_column = target_column
            self._fitted = True

            logger.info("rolling_mean_model_fitted")

        except Exception as e:
            logger.error(
                "rolling_mean_fitting_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to fit rolling mean model: {e}")

    def predict(self, horizon: int) -> pl.DataFrame:
        """Generate predictions for future periods.

        All future predictions use the mean of the last 'window' observed values.

        Args:
            horizon: Number of periods to predict into the future.

        Returns:
            DataFrame with columns:
            - timestamp: Future timestamps
            - prediction: Predicted values (constant for all periods)

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If horizon is not positive.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        logger.info("generating_rolling_mean_predictions", horizon=horizon)

        try:
            assert self._training_data is not None

            # Calculate mean of last 'window' values
            last_values = self._training_data[self._target_column].tail(self.window)
            prediction_value = float(last_values.mean())

            # Get last timestamp
            last_timestamp = self._training_data["timestamp"].max()

            # Infer frequency from training data
            if self._training_data.height >= 2:
                time_diff = (
                    self._training_data["timestamp"][1] - self._training_data["timestamp"][0]
                )
            else:
                # Default to 1 day
                time_diff = timedelta(days=1)

            # Create future timestamps
            future_timestamps = [
                last_timestamp + time_diff * (i + 1) for i in range(horizon)
            ]

            # All predictions are the same (rolling mean)
            predictions_df = pl.DataFrame(
                {
                    "timestamp": future_timestamps,
                    "prediction": [prediction_value] * horizon,
                }
            )

            logger.debug(
                "rolling_mean_predictions_generated",
                horizon=horizon,
                prediction_value=prediction_value,
            )

            return predictions_df

        except Exception as e:
            logger.error(
                "rolling_mean_prediction_failed",
                error=str(e),
                horizon=horizon,
                exc_info=True,
            )
            raise RuntimeError(f"Failed to generate predictions: {e}")

    def evaluate(self, test_df: pl.DataFrame) -> dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            test_df: DataFrame with actual values for comparison.

        Returns:
            Dictionary of evaluation metrics (RMSE, MAE, MAPE, direction_accuracy).

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If test_df is empty or missing required columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation. Call fit() first.")

        if test_df.height == 0:
            raise ValueError("Cannot evaluate on empty DataFrame")

        if self._target_column not in test_df.columns:
            raise ValueError(
                f"Target column '{self._target_column}' not found in test DataFrame"
            )

        logger.info("evaluating_rolling_mean_model", test_samples=test_df.height)

        try:
            # Sort test data
            test_sorted = test_df.sort("timestamp")

            # Generate predictions
            horizon = test_sorted.height
            predictions_df = self.predict(horizon=horizon)

            # Extract actual and predicted values
            actual = test_sorted[self._target_column]
            predicted = predictions_df["prediction"]

            # Calculate metrics
            metrics = calculate_metrics(actual, predicted)

            logger.info(
                "rolling_mean_evaluation_complete",
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                mape=metrics["mape"],
                direction_accuracy=metrics["direction_accuracy"],
            )

            # Log metrics to MLflow if in active run
            try:
                log_metrics(metrics)
            except RuntimeError:
                # No active MLflow run, skip logging
                logger.debug("no_active_mlflow_run", action="skipping metrics logging")

            return metrics

        except Exception as e:
            logger.error(
                "rolling_mean_evaluation_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to evaluate model: {e}")

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained.

        Returns:
            True if model is fitted and ready for prediction.
        """
        return self._fitted and self._training_data is not None
