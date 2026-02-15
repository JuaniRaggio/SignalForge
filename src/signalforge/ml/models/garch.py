"""GARCH model for volatility forecasting.

This module implements the Generalized Autoregressive Conditional Heteroskedasticity
(GARCH) model for forecasting volatility in financial time series.

GARCH models are used to:
- Model and forecast volatility clustering
- Estimate conditional variance
- Risk management and Value-at-Risk calculations
- Option pricing

The implementation supports various GARCH variants including standard GARCH,
EGARCH, and TGARCH, with different error distributions.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from arch import arch_model

from signalforge.core.logging import get_logger
from signalforge.ml.models.base import BasePredictor, PredictionResult

logger = get_logger(__name__)


class GARCHPredictor(BasePredictor):
    """GARCH model for volatility forecasting.

    Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models
    the variance of returns as a function of past variances and past squared returns.

    This is particularly useful for financial time series where volatility changes
    over time (heteroskedasticity) and shows clustering behavior.

    Attributes:
        model_name: Name identifier for the model
        model_version: Version string for the model
        p: GARCH order for lagged variance terms
        q: GARCH order for lagged residual terms
        mean: Mean model specification
        vol: Volatility model specification
        dist: Error distribution
        _model: Fitted GARCH model
        _fitted_result: Result object from model fitting
        _training_timestamps: Timestamps from training data
        _fitted: Boolean flag indicating if model is trained

    Examples:
        Basic usage:

        >>> import polars as pl
        >>> import numpy as np
        >>> from datetime import datetime
        >>> from signalforge.ml.models.garch import GARCHPredictor
        >>>
        >>> # Generate returns data
        >>> X = pl.DataFrame({
        ...     "timestamp": pl.date_range(
        ...         start=datetime(2024, 1, 1),
        ...         end=datetime(2024, 4, 9),
        ...         interval="1d"
        ...     ),
        ... })
        >>> returns = pl.Series(np.random.randn(100) * 0.02)  # 2% daily volatility
        >>> model = GARCHPredictor(p=1, q=1)
        >>> model.fit(X, returns)
        >>> predictions = model.predict(X.head(5))
    """

    model_name = "garch"
    model_version = "1.0.0"

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        vol: str = "GARCH",
        dist: str = "normal",
    ) -> None:
        """Initialize GARCH predictor.

        Args:
            p: GARCH order for lagged variance terms. Defaults to 1.
            q: GARCH order for lagged residual terms. Defaults to 1.
            mean: Mean model ("Constant", "Zero", "AR", "ARX"). Defaults to "Constant".
            vol: Volatility model ("GARCH", "EGARCH", "TGARCH"). Defaults to "GARCH".
            dist: Error distribution ("normal", "t", "skewt", "ged"). Defaults to "normal".

        Raises:
            ValueError: If p or q are not positive, or if mean/vol/dist are invalid.
        """
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if q <= 0:
            raise ValueError(f"q must be positive, got {q}")

        valid_means = ["Constant", "Zero", "AR", "ARX", "HAR", "LS"]
        if mean not in valid_means:
            raise ValueError(f"mean must be one of {valid_means}, got {mean}")

        valid_vols = ["GARCH", "EGARCH", "TGARCH", "FIGARCH", "HARCH"]
        if vol not in valid_vols:
            raise ValueError(f"vol must be one of {valid_vols}, got {vol}")

        valid_dists = ["normal", "t", "skewt", "ged"]
        if dist not in valid_dists:
            raise ValueError(f"dist must be one of {valid_dists}, got {dist}")

        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self._model: Any = None
        self._fitted_result: Any = None
        self._training_timestamps: list[datetime] | None = None
        self._fitted: bool = False

        logger.debug(
            "garch_predictor_initialized",
            p=p,
            q=q,
            mean=mean,
            vol=vol,
            dist=dist,
        )

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> GARCHPredictor:
        """Fit GARCH model on returns.

        GARCH models are typically fit on returns (not prices). The input series y
        should contain returns data.

        Args:
            X: Feature matrix with timestamp column
            y: Target series containing returns
            **kwargs: Additional arguments (e.g., update_freq for optimization verbosity)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If X or y is empty or invalid
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

        # Convert to numpy and scale to percentage returns (arch expects percentage)
        returns_array = y_sorted.to_numpy() * 100.0

        logger.info(
            "fitting_garch_model",
            n_samples=len(returns_array),
            p=self.p,
            q=self.q,
            mean=self.mean,
            vol=self.vol,
            dist=self.dist,
        )

        try:
            # Create GARCH model
            # Cast to proper literals for arch_model
            mean_lit = cast(Literal["Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"], self.mean)
            vol_lit = cast(Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"], self.vol)
            dist_lit = cast(Literal["normal", "gaussian", "t", "studentst", "skewstudent", "skewt", "ged", "generalized error"], self.dist)

            self._model = arch_model(
                returns_array,
                mean=mean_lit,
                vol=vol_lit,
                p=self.p,
                q=self.q,
                dist=dist_lit,
            )

            # Fit the model
            # Set update_freq to reduce verbosity (0 = silent)
            update_freq = kwargs.get("update_freq", 0)
            self._fitted_result = self._model.fit(update_freq=update_freq, disp="off")

            # Store training timestamps
            self._training_timestamps = X_sorted["timestamp"].to_list()
            self._fitted = True

            logger.info(
                "garch_model_fitted",
                aic=float(self._fitted_result.aic),
                bic=float(self._fitted_result.bic),
            )

            return self

        except Exception as e:
            logger.error(
                "garch_fitting_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to fit GARCH model: {e}") from e

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Predict volatility for future periods.

        Args:
            X: Feature matrix with timestamp column for prediction periods

        Returns:
            List of PredictionResult objects with volatility forecasts

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If X is empty or invalid
        """
        if not self._fitted or self._fitted_result is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        if X.height == 0:
            raise ValueError("Cannot predict on empty DataFrame")
        if "timestamp" not in X.columns:
            raise ValueError("X must contain 'timestamp' column")

        logger.info("generating_garch_predictions", n_periods=X.height)

        try:
            # Forecast volatility
            horizon = X.height
            forecast = self._fitted_result.forecast(horizon=horizon)

            # Extract variance forecast and convert to volatility (std dev)
            # arch returns percentage variance, convert back to decimal volatility
            variance_forecast = forecast.variance.values[-1, :]
            volatility_forecast = np.sqrt(variance_forecast) / 100.0

            # Create PredictionResult objects
            results = []
            timestamps = X["timestamp"].to_list()

            for i in range(horizon):
                vol = float(volatility_forecast[i])

                # For volatility, confidence is harder to quantify
                # Use a fixed moderate confidence as GARCH is well-suited for this
                confidence = 0.75

                result = PredictionResult(
                    symbol="UNKNOWN",  # Will be set by caller if needed
                    timestamp=timestamps[i],
                    horizon_days=i + 1,
                    prediction=vol,
                    confidence=confidence,
                    lower_bound=None,  # GARCH doesn't provide standard intervals
                    upper_bound=None,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
                results.append(result)

            logger.debug(
                "garch_predictions_generated",
                n_predictions=len(results),
                first_prediction=results[0].prediction if results else None,
            )

            return results

        except Exception as e:
            logger.error(
                "garch_prediction_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to generate predictions: {e}") from e

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return volatility forecast with variance.

        Args:
            X: Feature matrix with timestamp column

        Returns:
            DataFrame with columns: timestamp, volatility, variance

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If X is invalid
        """
        if not self._fitted or self._fitted_result is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        horizon = X.height
        forecast = self._fitted_result.forecast(horizon=horizon)

        # Extract variance and convert to volatility
        variance_forecast = forecast.variance.values[-1, :] / 10000.0  # Convert from percentage^2
        volatility_forecast = np.sqrt(variance_forecast)

        return pl.DataFrame({
            "timestamp": X["timestamp"],
            "volatility": volatility_forecast,
            "variance": variance_forecast,
        })

    def forecast_variance(self, horizon: int = 5) -> pl.DataFrame:
        """Forecast conditional variance for specified horizon.

        Args:
            horizon: Number of periods to forecast. Defaults to 5.

        Returns:
            DataFrame with columns: horizon_step, variance, volatility

        Raises:
            RuntimeError: If model has not been fitted
            ValueError: If horizon is not positive
        """
        if not self._fitted or self._fitted_result is None:
            raise RuntimeError("Model must be fitted before forecasting variance")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        forecast = self._fitted_result.forecast(horizon=horizon)

        # Extract variance forecast (convert from percentage^2 to decimal^2)
        variance_forecast = forecast.variance.values[-1, :] / 10000.0
        volatility_forecast = np.sqrt(variance_forecast)

        return pl.DataFrame({
            "horizon_step": list(range(1, horizon + 1)),
            "variance": variance_forecast,
            "volatility": volatility_forecast,
        })

    def get_standardized_residuals(self) -> pl.Series:
        """Return standardized residuals for diagnostics.

        Standardized residuals should be approximately N(0,1) if the model
        is correctly specified.

        Returns:
            Series of standardized residuals

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._fitted or self._fitted_result is None:
            raise RuntimeError("Model must be fitted before getting residuals")

        std_resids = self._fitted_result.std_resid
        return pl.Series(std_resids)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path where the model should be saved

        Raises:
            RuntimeError: If model has not been fitted
            IOError: If saving fails
        """
        if not self._fitted or self._fitted_result is None:
            raise RuntimeError("Cannot save unfitted model")

        logger.info("saving_garch_model", path=path)

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                "p": self.p,
                "q": self.q,
                "mean": self.mean,
                "vol": self.vol,
                "dist": self.dist,
                "fitted_result": self._fitted_result,
                "training_timestamps": self._training_timestamps,
                "model_name": self.model_name,
                "model_version": self.model_version,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info("garch_model_saved", path=path)

        except Exception as e:
            logger.error("garch_save_failed", error=str(e), exc_info=True)
            raise OSError(f"Failed to save model: {e}") from e

    @classmethod
    def load(cls, path: str) -> GARCHPredictor:
        """Load model from disk.

        Args:
            path: File path from which to load the model

        Returns:
            Loaded GARCHPredictor instance

        Raises:
            IOError: If loading fails
            ValueError: If file contains invalid model data
        """
        logger.info("loading_garch_model", path=path)

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            predictor = cls(
                p=model_data["p"],
                q=model_data["q"],
                mean=model_data["mean"],
                vol=model_data["vol"],
                dist=model_data["dist"],
            )
            predictor._fitted_result = model_data["fitted_result"]
            predictor._training_timestamps = model_data["training_timestamps"]
            predictor._fitted = True
            predictor.model_name = model_data.get("model_name", cls.model_name)
            predictor.model_version = model_data.get("model_version", cls.model_version)

            logger.info("garch_model_loaded", path=path)

            return predictor

        except Exception as e:
            logger.error("garch_load_failed", error=str(e), exc_info=True)
            raise OSError(f"Failed to load model: {e}") from e
