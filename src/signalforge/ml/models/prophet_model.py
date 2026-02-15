"""Facebook Prophet model for time series with seasonality.

This module implements the Prophet forecasting model, which is designed for
business time series with strong seasonal patterns and multiple seasonality.

Prophet is particularly good at:
- Handling missing data
- Trend changes
- Outliers
- Multiple seasonality (daily, weekly, yearly)
- Holiday effects

The model decomposes time series into trend, seasonality, and holidays.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import polars as pl
from prophet import Prophet

from signalforge.core.logging import get_logger
from signalforge.ml.models.base import BasePredictor, PredictionResult

logger = get_logger(__name__)


class ProphetPredictor(BasePredictor):
    """Facebook Prophet model for time series with seasonality.

    Prophet is a procedure for forecasting time series data based on an additive
    model where non-linear trends are fit with yearly, weekly, and daily seasonality,
    plus holiday effects.

    Attributes:
        model_name: Name identifier for the model
        model_version: Version string for the model
        yearly_seasonality: Enable yearly seasonal component
        weekly_seasonality: Enable weekly seasonal component
        daily_seasonality: Enable daily seasonal component
        changepoint_prior_scale: Flexibility of trend (higher = more flexible)
        _model: Fitted Prophet model
        _fitted: Boolean flag indicating if model is trained

    Examples:
        Basic usage:

        >>> import polars as pl
        >>> from datetime import datetime
        >>> from signalforge.ml.models.prophet_model import ProphetPredictor
        >>>
        >>> X = pl.DataFrame({
        ...     "timestamp": pl.date_range(
        ...         start=datetime(2024, 1, 1),
        ...         end=datetime(2024, 4, 9),
        ...         interval="1d"
        ...     ),
        ... })
        >>> y = pl.Series([100.0 + i * 0.5 for i in range(100)])
        >>> model = ProphetPredictor()
        >>> model.fit(X, y)
        >>> predictions = model.predict(X.head(10))
    """

    model_name = "prophet"
    model_version = "1.0.0"

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ) -> None:
        """Initialize Prophet predictor.

        Args:
            yearly_seasonality: Enable yearly seasonal component. Defaults to True.
            weekly_seasonality: Enable weekly seasonal component. Defaults to True.
            daily_seasonality: Enable daily seasonal component. Defaults to False.
            changepoint_prior_scale: Flexibility of trend. Higher values allow
                                     more flexible trend. Defaults to 0.05.

        Raises:
            ValueError: If changepoint_prior_scale is not positive.
        """
        if changepoint_prior_scale <= 0:
            raise ValueError(f"changepoint_prior_scale must be positive, got {changepoint_prior_scale}")

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._model: Prophet | None = None
        self._fitted: bool = False
        self._regressors: list[str] = []

        logger.debug(
            "prophet_predictor_initialized",
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
        )

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> ProphetPredictor:  # noqa: ARG002
        """Fit Prophet model.

        Prophet requires data in a specific format with 'ds' (datestamp) and 'y' (target).
        This method transforms the input to Prophet's expected format.

        Args:
            X: Feature matrix with timestamp column
            y: Target series containing the values to predict
            **kwargs: Additional arguments (unused for Prophet)

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

        logger.info(
            "fitting_prophet_model",
            n_samples=len(y),
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )

        try:
            # Create Prophet model with specified parameters
            # Suppress Prophet's cmdstanpy logger output
            self._model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
            )

            # Add any previously configured regressors
            for regressor in self._regressors:
                if regressor in X.columns:
                    self._model.add_regressor(regressor)

            # Transform data to Prophet format (requires 'ds' and 'y' columns)
            prophet_df = pl.DataFrame({
                "ds": X["timestamp"],
                "y": y,
            })

            # Add regressor columns if they exist
            for regressor in self._regressors:
                if regressor in X.columns:
                    prophet_df = prophet_df.with_columns(X[regressor].alias(regressor))

            # Convert to pandas for Prophet (it requires pandas)
            prophet_df_pandas = prophet_df.to_pandas()

            # Fit the model
            self._model.fit(prophet_df_pandas)
            self._fitted = True

            logger.info("prophet_model_fitted")

            return self

        except Exception as e:
            logger.error(
                "prophet_fitting_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to fit Prophet model: {e}") from e

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
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        if X.height == 0:
            raise ValueError("Cannot predict on empty DataFrame")
        if "timestamp" not in X.columns:
            raise ValueError("X must contain 'timestamp' column")

        logger.info("generating_prophet_predictions", n_periods=X.height)

        try:
            # Create future dataframe in Prophet format
            future_df = pl.DataFrame({"ds": X["timestamp"]})

            # Add regressor columns if they exist
            for regressor in self._regressors:
                if regressor in X.columns:
                    future_df = future_df.with_columns(X[regressor].alias(regressor))

            # Convert to pandas for Prophet
            future_df_pandas = future_df.to_pandas()

            # Generate predictions
            forecast = self._model.predict(future_df_pandas)

            # Create PredictionResult objects
            results = []
            timestamps = X["timestamp"].to_list()

            for i in range(len(forecast)):
                # Prophet provides yhat (prediction), yhat_lower, yhat_upper
                yhat = float(forecast["yhat"].iloc[i])
                yhat_lower = float(forecast["yhat_lower"].iloc[i])
                yhat_upper = float(forecast["yhat_upper"].iloc[i])

                # Calculate confidence based on uncertainty interval width
                interval_width = yhat_upper - yhat_lower
                # Use a simple heuristic: narrower intervals = higher confidence
                confidence = max(0.0, min(1.0, 1.0 - (interval_width / (2 * abs(yhat) + 1e-10))))

                result = PredictionResult(
                    symbol="UNKNOWN",  # Will be set by caller if needed
                    timestamp=timestamps[i],
                    horizon_days=i + 1,
                    prediction=yhat,
                    confidence=confidence,
                    lower_bound=yhat_lower,
                    upper_bound=yhat_upper,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
                results.append(result)

            logger.debug(
                "prophet_predictions_generated",
                n_predictions=len(results),
                first_prediction=results[0].prediction if results else None,
            )

            return results

        except Exception as e:
            logger.error(
                "prophet_prediction_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to generate predictions: {e}") from e

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return predictions with uncertainty intervals.

        Prophet provides uncertainty intervals (yhat_lower, yhat_upper) which
        represent the expected range of values.

        Args:
            X: Feature matrix with timestamp column

        Returns:
            DataFrame with columns: timestamp, prediction, yhat_lower, yhat_upper, confidence

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If X is invalid
        """
        predictions = self.predict(X)

        return pl.DataFrame({
            "timestamp": [p.timestamp for p in predictions],
            "prediction": [p.prediction for p in predictions],
            "yhat_lower": [p.lower_bound for p in predictions],
            "yhat_upper": [p.upper_bound for p in predictions],
            "confidence": [p.confidence for p in predictions],
        })

    def add_regressor(self, name: str) -> None:
        """Add external regressor to the model.

        Regressors are additional features that can help improve predictions.
        They must be provided during both fit() and predict().

        Args:
            name: Name of the regressor column (must exist in X during fit/predict)

        Raises:
            RuntimeError: If model has already been fitted
        """
        if self._fitted:
            raise RuntimeError("Cannot add regressors after model has been fitted")

        self._regressors.append(name)
        logger.debug("prophet_regressor_added", name=name)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path where the model should be saved

        Raises:
            RuntimeError: If model has not been fitted
            IOError: If saving fails
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Cannot save unfitted model")

        logger.info("saving_prophet_model", path=path)

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                "model": self._model,
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality,
                "daily_seasonality": self.daily_seasonality,
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "regressors": self._regressors,
                "model_name": self.model_name,
                "model_version": self.model_version,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info("prophet_model_saved", path=path)

        except Exception as e:
            logger.error("prophet_save_failed", error=str(e), exc_info=True)
            raise OSError(f"Failed to save model: {e}") from e

    @classmethod
    def load(cls, path: str) -> ProphetPredictor:
        """Load model from disk.

        Args:
            path: File path from which to load the model

        Returns:
            Loaded ProphetPredictor instance

        Raises:
            IOError: If loading fails
            ValueError: If file contains invalid model data
        """
        logger.info("loading_prophet_model", path=path)

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            predictor = cls(
                yearly_seasonality=model_data["yearly_seasonality"],
                weekly_seasonality=model_data["weekly_seasonality"],
                daily_seasonality=model_data["daily_seasonality"],
                changepoint_prior_scale=model_data["changepoint_prior_scale"],
            )
            predictor._model = model_data["model"]
            predictor._regressors = model_data.get("regressors", [])
            predictor._fitted = True
            predictor.model_name = model_data.get("model_name", cls.model_name)
            predictor.model_version = model_data.get("model_version", cls.model_version)

            logger.info("prophet_model_loaded", path=path)

            return predictor

        except Exception as e:
            logger.error("prophet_load_failed", error=str(e), exc_info=True)
            raise OSError(f"Failed to load model: {e}") from e
