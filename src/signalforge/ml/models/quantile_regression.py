"""Quantile regression for prediction intervals.

This module implements quantile regression models that estimate conditional
quantiles of the target variable, enabling prediction intervals and uncertainty
quantification. This is crucial for risk assessment in financial forecasting.

The QuantileRegressionPredictor properly implements the BasePredictor interface
with fit(X, y) and predict(X) methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import structlog
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor as SKLearnQuantileRegressor

from signalforge.ml.models.base import BasePredictor, PredictionResult

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class QuantileRegressionConfig(BaseModel):
    """Configuration for quantile regression models."""

    quantiles: list[float] = Field(default=[0.1, 0.25, 0.5, 0.75, 0.9])
    base_model: Literal["linear", "gradient_boosting"] = "gradient_boosting"
    n_estimators: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=0.1, gt=0)
    max_depth: int = Field(default=3, gt=0)
    alpha: float = Field(default=1.0, gt=0)


@dataclass
class QuantilePrediction:
    """Result of a quantile prediction."""

    prediction: float
    quantile_values: dict[float, float] = field(default_factory=dict)
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    confidence: float = 0.0


# Alias for gradient boosting variant
QuantileGradientBoostingRegressor = GradientBoostingRegressor


# Alias for linear variant
QuantileRegressor = SKLearnQuantileRegressor


def calculate_coverage(
    y_actual: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Calculate empirical coverage of prediction intervals.

    Args:
        y_actual: Actual observed values.
        lower: Lower bounds of intervals.
        upper: Upper bounds of intervals.

    Returns:
        Coverage rate (0.0 to 1.0).
    """
    in_interval = (y_actual >= lower) & (y_actual <= upper)
    return float(np.mean(in_interval))


def winkler_score(
    y_actual: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    alpha: float = 0.1,
) -> float:
    """Calculate Winkler score for prediction intervals.

    The Winkler score penalizes both interval width and coverage misses.

    Args:
        y_actual: Actual observed values.
        lower: Lower bounds of intervals.
        upper: Upper bounds of intervals.
        alpha: Significance level (default: 0.1 for 90% intervals).

    Returns:
        Average Winkler score (lower is better).
    """
    interval_width = upper - lower
    below_lower = y_actual < lower
    above_upper = y_actual > upper

    score = interval_width.copy()
    score[below_lower] += (2.0 / alpha) * (lower[below_lower] - y_actual[below_lower])
    score[above_upper] += (2.0 / alpha) * (y_actual[above_upper] - upper[above_upper])

    return float(np.mean(score))


def create_quantile_regressor(
    config: QuantileRegressionConfig | None = None,
) -> QuantileRegressionPredictor:
    """Factory function to create a quantile regression predictor.

    Args:
        config: Configuration for the model.

    Returns:
        Configured QuantileRegressionPredictor.
    """
    if config is None:
        config = QuantileRegressionConfig()

    return QuantileRegressionPredictor(
        quantiles=config.quantiles,
        base_model=config.base_model,
    )


class QuantileRegressionPredictor(BasePredictor):
    """Quantile regression for prediction intervals.

    This model fits separate quantile regressors for each specified quantile level,
    enabling the generation of prediction intervals rather than point forecasts.
    It supports both linear quantile regression and gradient boosting.

    Quantile regression is particularly useful in finance where understanding
    the full distribution of possible outcomes is more valuable than just
    the expected value.

    Attributes:
        model_name: Human-readable name of the model.
        model_version: Version string for the model.

    Examples:
        Basic usage with default quantiles:

        >>> import polars as pl
        >>> from signalforge.ml.models.quantile_regression import (
        ...     QuantileRegressionPredictor
        ... )
        >>>
        >>> model = QuantileRegressionPredictor(
        ...     quantiles=[0.1, 0.5, 0.9],
        ...     base_model="linear"
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> intervals = model.predict_intervals(X_test, confidence=0.8)

        With gradient boosting:

        >>> model = QuantileRegressionPredictor(
        ...     quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        ...     base_model="gradient_boosting"
        ... )
    """

    model_name: str = "quantile_regression"
    model_version: str = "1.0.0"

    def __init__(
        self,
        quantiles: list[float] | None = None,
        base_model: Literal["linear", "gradient_boosting"] = "gradient_boosting",
    ) -> None:
        """Initialize quantile regression predictor.

        Args:
            quantiles: List of quantile levels to estimate (e.g., [0.1, 0.5, 0.9]).
                      Values must be in (0, 1). Should include 0.5 for median.
                      Default: [0.1, 0.25, 0.5, 0.75, 0.9]
            base_model: Type of base model to use:
                - "linear": Linear quantile regression (fast, interpretable)
                - "gradient_boosting": Gradient boosting (handles non-linearity)

        Raises:
            ValueError: If quantiles are invalid or empty.
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        if not quantiles:
            raise ValueError("At least one quantile must be specified")

        for q in quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantile {q} must be in range (0, 1)")

        if len(quantiles) != len(set(quantiles)):
            raise ValueError("Quantiles must be unique")

        self.quantiles = sorted(quantiles)
        self.base_model = base_model
        self._models: dict[float, Any] = {}
        self._fitted = False
        self._feature_columns: list[str] = []

        logger.info(
            "quantile_regression_initialized",
            quantiles=self.quantiles,
            base_model=base_model,
        )

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> QuantileRegressionPredictor:
        """Fit quantile regression models for each quantile.

        Trains separate models for each quantile level on the provided data.

        Args:
            X: Feature matrix with historical data.
            y: Target series containing values to predict.
            **kwargs: Additional parameters for gradient boosting:
                - n_estimators: int, number of boosting stages (default: 100)
                - learning_rate: float, learning rate (default: 0.1)
                - max_depth: int, maximum tree depth (default: 3)
                - alpha: float, L1 regularization for linear (default: 1.0)

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X and y have incompatible shapes.
            RuntimeError: If model fitting fails.
        """
        if X.height != len(y):
            raise ValueError(
                f"X and y must have same length: X={X.height}, y={len(y)}"
            )

        if X.height == 0:
            raise ValueError("Cannot fit on empty data")

        logger.info(
            "fitting_quantile_regression",
            n_samples=X.height,
            n_features=len(X.columns),
            quantiles=self.quantiles,
        )

        self._feature_columns = X.columns

        try:
            X_np = X.to_numpy()
            y_np = y.to_numpy()

            # Fit model for each quantile
            for quantile in self.quantiles:
                logger.debug(f"fitting_quantile_{quantile}")

                if self.base_model == "linear":
                    alpha = kwargs.get("alpha", 1.0)
                    model = SKLearnQuantileRegressor(
                        quantile=quantile,
                        alpha=alpha,
                        solver="highs",
                    )
                else:  # gradient_boosting
                    n_estimators = kwargs.get("n_estimators", 100)
                    learning_rate = kwargs.get("learning_rate", 0.1)
                    max_depth = kwargs.get("max_depth", 3)

                    model = GradientBoostingRegressor(
                        loss="quantile",
                        alpha=quantile,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42,
                    )

                model.fit(X_np, y_np)
                self._models[quantile] = model
                logger.debug(f"quantile_{quantile}_fitted")

            self._fitted = True
            logger.info("quantile_regression_fitted_successfully")

            return self

        except Exception as e:
            logger.error("quantile_regression_fitting_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to fit quantile regression: {e}") from e

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Generate predictions with intervals from quantiles.

        Predicts all quantile levels and returns results with prediction intervals
        derived from the lowest and highest quantiles.

        Args:
            X: Feature matrix for which to generate predictions.

        Returns:
            List of PredictionResult objects with prediction intervals.
            The prediction field contains the median (0.5 quantile) if available,
            otherwise the mean of all quantiles.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has incompatible features.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if set(X.columns) != set(self._feature_columns):
            raise ValueError(
                f"Feature mismatch. Expected {self._feature_columns}, "
                f"got {X.columns}"
            )

        logger.info("generating_quantile_predictions", n_samples=X.height)

        try:
            X_np = X.to_numpy()

            # Get predictions for all quantiles
            quantile_preds: dict[float, NDArray[np.float64]] = {}
            for quantile in self.quantiles:
                quantile_preds[quantile] = self._models[quantile].predict(X_np)

            # Build PredictionResults
            results: list[PredictionResult] = []

            for i in range(X.height):
                # Extract predictions for this sample
                sample_quantiles = {q: float(quantile_preds[q][i]) for q in self.quantiles}

                # Use median as point prediction if available, else mean
                if 0.5 in sample_quantiles:
                    prediction = sample_quantiles[0.5]
                else:
                    prediction = float(np.mean(list(sample_quantiles.values())))

                # Bounds from min/max quantiles
                lower_bound = sample_quantiles[self.quantiles[0]]
                upper_bound = sample_quantiles[self.quantiles[-1]]

                # Confidence based on interval width
                # Narrower interval means higher confidence
                if upper_bound != lower_bound:
                    interval_width = upper_bound - lower_bound
                    # Normalize to 0-1 scale (inverse relationship)
                    confidence = float(1.0 / (1.0 + interval_width / abs(prediction + 1e-8)))
                else:
                    confidence = 1.0

                result = PredictionResult(
                    symbol="",
                    timestamp=datetime.now(),
                    horizon_days=1,
                    prediction=prediction,
                    confidence=confidence,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
                results.append(result)

            logger.info("quantile_predictions_generated", n_predictions=len(results))
            return results

        except Exception as e:
            logger.error("quantile_prediction_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to generate predictions: {e}") from e

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return predictions for all quantiles.

        Provides the full quantile distribution for each sample, enabling
        comprehensive uncertainty analysis.

        Args:
            X: Feature matrix for which to generate probability predictions.

        Returns:
            DataFrame with columns:
            - prediction: Point forecast (median or mean)
            - lower_bound: Lowest quantile prediction
            - upper_bound: Highest quantile prediction
            - confidence: Confidence score
            - quantile_<q>: Prediction for each quantile q

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has incompatible features.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        logger.info("generating_quantile_proba", n_samples=X.height)

        X_np = X.to_numpy()

        # Get predictions for all quantiles
        quantile_preds: dict[float, NDArray[np.float64]] = {}
        for quantile in self.quantiles:
            quantile_preds[quantile] = self._models[quantile].predict(X_np)

        # Build DataFrame
        rows: list[dict[str, Any]] = []

        for i in range(X.height):
            sample_quantiles = {q: float(quantile_preds[q][i]) for q in self.quantiles}

            # Point prediction
            if 0.5 in sample_quantiles:
                prediction = sample_quantiles[0.5]
            else:
                prediction = float(np.mean(list(sample_quantiles.values())))

            # Bounds
            lower_bound = sample_quantiles[self.quantiles[0]]
            upper_bound = sample_quantiles[self.quantiles[-1]]

            # Confidence
            if upper_bound != lower_bound:
                interval_width = upper_bound - lower_bound
                confidence = float(1.0 / (1.0 + interval_width / abs(prediction + 1e-8)))
            else:
                confidence = 1.0

            row: dict[str, Any] = {
                "prediction": prediction,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence": confidence,
            }

            # Add individual quantile predictions
            for q, val in sample_quantiles.items():
                row[f"quantile_{q}"] = val

            rows.append(row)

        result_df = pl.DataFrame(rows)
        logger.info("quantile_proba_generated")
        return result_df

    def predict_intervals(
        self,
        X: pl.DataFrame,
        confidence: float = 0.8,
    ) -> pl.DataFrame:
        """Return prediction intervals at specified confidence level.

        Constructs prediction intervals from the quantile predictions that
        match the requested confidence level.

        Args:
            X: Feature matrix for which to generate intervals.
            confidence: Desired confidence level (e.g., 0.8 for 80% interval).
                       Must be in (0, 1).

        Returns:
            DataFrame with columns:
            - prediction: Point forecast
            - lower_bound: Lower interval bound
            - upper_bound: Upper interval bound
            - confidence: Actual confidence level
            - interval_width: Width of the prediction interval

        Raises:
            RuntimeError: If model not fitted.
            ValueError: If confidence is invalid or no quantiles match.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

        logger.info("generating_prediction_intervals", confidence=confidence, n_samples=X.height)

        # Find quantiles that correspond to confidence level
        # For 80% confidence, use 0.1 and 0.9 quantiles
        alpha = (1.0 - confidence) / 2.0
        lower_quantile = alpha
        upper_quantile = 1.0 - alpha

        # Find closest available quantiles
        lower_q = min(self.quantiles, key=lambda q: abs(q - lower_quantile))
        upper_q = min(self.quantiles, key=lambda q: abs(q - upper_quantile))

        actual_confidence = upper_q - lower_q

        logger.info(
            "using_quantiles_for_interval",
            lower_q=lower_q,
            upper_q=upper_q,
            actual_confidence=actual_confidence,
        )

        X_np = X.to_numpy()

        # Get predictions
        lower_preds = self._models[lower_q].predict(X_np)
        upper_preds = self._models[upper_q].predict(X_np)

        # Get point predictions
        if 0.5 in self._models:
            point_preds = self._models[0.5].predict(X_np)
        else:
            # Average all quantiles
            all_preds = [self._models[q].predict(X_np) for q in self.quantiles]
            point_preds = np.mean(all_preds, axis=0)

        # Build DataFrame
        rows: list[dict[str, Any]] = []
        for i in range(X.height):
            rows.append({
                "prediction": float(point_preds[i]),
                "lower_bound": float(lower_preds[i]),
                "upper_bound": float(upper_preds[i]),
                "confidence": actual_confidence,
                "interval_width": float(upper_preds[i] - lower_preds[i]),
            })

        result_df = pl.DataFrame(rows)
        logger.info("prediction_intervals_generated")
        return result_df

    def calibrate_intervals(
        self,
        X: pl.DataFrame,
        y: pl.Series,
    ) -> dict[float, float]:
        """Check calibration: does 80% interval contain 80% of actuals?

        Evaluates whether the prediction intervals are well-calibrated by
        computing empirical coverage for different confidence levels.

        Args:
            X: Feature matrix.
            y: Actual observed values.

        Returns:
            Dictionary mapping confidence levels to empirical coverage rates.
            Well-calibrated models should have empirical coverage close to
            the nominal confidence level.

        Raises:
            RuntimeError: If model not fitted.
            ValueError: If X and y length mismatch.

        Examples:
            >>> calibration = model.calibrate_intervals(X_val, y_val)
            >>> print(f"80% interval coverage: {calibration[0.8]:.2%}")
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calibration")

        if X.height != len(y):
            raise ValueError(f"X and y length mismatch: {X.height} vs {len(y)}")

        logger.info("calibrating_intervals", n_samples=X.height)

        y_actual = y.to_numpy()

        # Test various confidence levels
        confidence_levels = [0.5, 0.68, 0.8, 0.9, 0.95]
        calibration: dict[float, float] = {}

        for conf_level in confidence_levels:
            try:
                intervals_df = self.predict_intervals(X, confidence=conf_level)

                lower = intervals_df["lower_bound"].to_numpy()
                upper = intervals_df["upper_bound"].to_numpy()

                # Calculate empirical coverage
                in_interval = (y_actual >= lower) & (y_actual <= upper)
                empirical_coverage = float(np.mean(in_interval))

                calibration[conf_level] = empirical_coverage

                logger.debug(
                    "confidence_level_calibration",
                    nominal=conf_level,
                    empirical=empirical_coverage,
                    deviation=abs(empirical_coverage - conf_level),
                )

            except ValueError:
                # Skip if quantiles not available for this confidence level
                continue

        logger.info("interval_calibration_completed", results=calibration)
        return calibration
