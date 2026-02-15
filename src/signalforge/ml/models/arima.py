"""ARIMA model for time series prediction.

This module implements the AutoRegressive Integrated Moving Average (ARIMA)
model using statsmodels, following the BasePredictor interface.

ARIMA combines:
- AR (AutoRegressive): Uses past values to predict future
- I (Integrated): Differencing to make series stationary
- MA (Moving Average): Uses past forecast errors

The model supports both non-seasonal and seasonal ARIMA (SARIMA).
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from statsmodels.tsa.arima.model import ARIMA

from signalforge.core.logging import get_logger
from signalforge.ml.models.base import BasePredictor, PredictionResult

logger = get_logger(__name__)


class ARIMAPredictor(BasePredictor):
    """ARIMA model for time series prediction.

    AutoRegressive Integrated Moving Average (ARIMA) is a classical statistical
    model for time series forecasting. This implementation wraps statsmodels ARIMA
    with the BasePredictor interface.

    Attributes:
        model_name: Name identifier for the model
        model_version: Version string for the model
        order: ARIMA order (p, d, q) where:
               p = number of lag observations
               d = degree of differencing
               q = size of moving average window
        seasonal_order: Optional seasonal ARIMA order (P, D, Q, s)
        _model: Fitted ARIMA model (None until fit() is called)
        _fitted_model: Result object from model fitting
        _training_timestamps: Timestamps from training data
        _fitted: Boolean flag indicating if model is trained

    Examples:
        Basic usage:

        >>> import polars as pl
        >>> from datetime import datetime
        >>> from signalforge.ml.models.arima import ARIMAPredictor
        >>>
        >>> X = pl.DataFrame({
        ...     "timestamp": pl.date_range(
        ...         start=datetime(2024, 1, 1),
        ...         end=datetime(2024, 4, 9),
        ...         interval="1d"
        ...     ),
        ... })
        >>> y = pl.Series([100.0 + i * 0.5 for i in range(100)])
        >>> model = ARIMAPredictor(order=(5, 1, 0))
        >>> model.fit(X, y)
        >>> predictions = model.predict(X.head(5))
    """

    model_name = "arima"
    model_version = "1.0.0"

    def __init__(
        self,
        order: tuple[int, int, int] = (5, 1, 0),
        seasonal_order: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Initialize ARIMA predictor.

        Args:
            order: ARIMA order (p, d, q). Defaults to (5, 1, 0).
            seasonal_order: Optional seasonal order (P, D, Q, s). If None, no seasonality.

        Raises:
            ValueError: If order parameters are invalid.
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
        self._training_timestamps: list[datetime] | None = None
        self._fitted: bool = False

        logger.debug(
            "arima_predictor_initialized",
            order=order,
            seasonal_order=seasonal_order,
        )

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> ARIMAPredictor:  # noqa: ARG002
        """Train the ARIMA model on historical data.

        Args:
            X: Feature matrix with timestamp column
            y: Target series containing the values to predict
            **kwargs: Additional arguments (unused for ARIMA)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If X or y is empty or missing required columns
            RuntimeError: If model fitting fails
        """
        if X.height == 0:
            raise ValueError("Cannot fit model on empty DataFrame")
        if len(y) == 0:
            raise ValueError("Cannot fit model on empty target series")
        if X.height != len(y):
            raise ValueError(f"X and y shape mismatch: {X.height} != {len(y)}")
        if "timestamp" not in X.columns:
            raise ValueError("X must contain 'timestamp' column")

        # Sort by timestamp to ensure chronological order
        sort_indices = X["timestamp"].arg_sort()
        X_sorted = X[sort_indices]
        y_sorted = y[sort_indices]

        # Extract target series as numpy array
        target_array = y_sorted.to_numpy()

        logger.info(
            "fitting_arima_model",
            n_samples=len(target_array),
            order=self.order,
            seasonal_order=self.seasonal_order,
        )

        try:
            # Fit ARIMA model
            if self.seasonal_order is not None:
                self._model = ARIMA(
                    target_array,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                )
            else:
                self._model = ARIMA(
                    target_array,
                    order=self.order,
                )
            self._fitted_model = self._model.fit()

            # Store training timestamps for prediction
            self._training_timestamps = X_sorted["timestamp"].to_list()
            self._fitted = True

            logger.info(
                "arima_model_fitted",
                aic=self._fitted_model.aic,
                bic=self._fitted_model.bic,
            )

            return self

        except Exception as e:
            logger.error(
                "arima_fitting_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to fit ARIMA model: {e}") from e

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Generate predictions for the given input features.

        Args:
            X: Feature matrix with timestamp column for prediction periods

        Returns:
            List of PredictionResult objects, one per row in X

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If X is empty or missing timestamp column
        """
        if not self._fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        if X.height == 0:
            raise ValueError("Cannot predict on empty DataFrame")
        if "timestamp" not in X.columns:
            raise ValueError("X must contain 'timestamp' column")

        logger.info("generating_arima_predictions", n_periods=X.height)

        try:
            # Get forecast for the required number of steps
            steps = X.height
            forecast = self._fitted_model.forecast(steps=steps)
            forecast_result = self._fitted_model.get_forecast(steps=steps)
            forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

            # Create PredictionResult objects
            results = []
            timestamps = X["timestamp"].to_list()

            for i in range(steps):
                # Calculate confidence as inverse of interval width (normalized)
                interval_width = forecast_ci[i, 1] - forecast_ci[i, 0]
                # Use a simple heuristic: narrower intervals = higher confidence
                # Normalize to 0-1 range (capped at reasonable values)
                confidence = max(0.0, min(1.0, 1.0 - (interval_width / (2 * abs(forecast[i]) + 1e-10))))

                result = PredictionResult(
                    symbol="UNKNOWN",  # Will be set by caller if needed
                    timestamp=timestamps[i],
                    horizon_days=i + 1,
                    prediction=float(forecast[i]),
                    confidence=float(confidence),
                    lower_bound=float(forecast_ci[i, 0]),
                    upper_bound=float(forecast_ci[i, 1]),
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
                results.append(result)

            logger.debug(
                "arima_predictions_generated",
                n_predictions=len(results),
                first_prediction=results[0].prediction if results else None,
            )

            return results

        except Exception as e:
            logger.error(
                "arima_prediction_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to generate predictions: {e}") from e

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return predictions with confidence intervals.

        Args:
            X: Feature matrix with timestamp column

        Returns:
            DataFrame with columns: timestamp, prediction, lower_bound, upper_bound, confidence

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If X is invalid
        """
        predictions = self.predict(X)

        return pl.DataFrame({
            "timestamp": [p.timestamp for p in predictions],
            "prediction": [p.prediction for p in predictions],
            "lower_bound": [p.lower_bound for p in predictions],
            "upper_bound": [p.upper_bound for p in predictions],
            "confidence": [p.confidence for p in predictions],
        })

    def get_aic(self) -> float:
        """Return Akaike Information Criterion for model selection.

        Returns:
            AIC value (lower is better)

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before getting AIC")
        return float(self._fitted_model.aic)

    def get_bic(self) -> float:
        """Return Bayesian Information Criterion for model selection.

        Returns:
            BIC value (lower is better)

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._fitted or self._fitted_model is None:
            raise RuntimeError("Model must be fitted before getting BIC")
        return float(self._fitted_model.bic)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path where the model should be saved

        Raises:
            RuntimeError: If model has not been fitted
            IOError: If saving fails
        """
        if not self._fitted or self._fitted_model is None:
            raise RuntimeError("Cannot save unfitted model")

        logger.info("saving_arima_model", path=path)

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "fitted_model": self._fitted_model,
                "training_timestamps": self._training_timestamps,
                "model_name": self.model_name,
                "model_version": self.model_version,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info("arima_model_saved", path=path)

        except Exception as e:
            logger.error("arima_save_failed", error=str(e), exc_info=True)
            raise OSError(f"Failed to save model: {e}") from e

    @classmethod
    def load(cls, path: str) -> ARIMAPredictor:
        """Load model from disk.

        Args:
            path: File path from which to load the model

        Returns:
            Loaded ARIMAPredictor instance

        Raises:
            IOError: If loading fails
            ValueError: If file contains invalid model data
        """
        logger.info("loading_arima_model", path=path)

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            predictor = cls(
                order=model_data["order"],
                seasonal_order=model_data["seasonal_order"],
            )
            predictor._fitted_model = model_data["fitted_model"]
            predictor._training_timestamps = model_data["training_timestamps"]
            predictor._fitted = True
            predictor.model_name = model_data.get("model_name", cls.model_name)
            predictor.model_version = model_data.get("model_version", cls.model_version)

            logger.info("arima_model_loaded", path=path)

            return predictor

        except Exception as e:
            logger.error("arima_load_failed", error=str(e), exc_info=True)
            raise OSError(f"Failed to load model: {e}") from e
