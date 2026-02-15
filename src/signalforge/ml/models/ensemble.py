"""Ensemble model combining multiple predictors.

This module implements an ensemble predictor that combines predictions from
multiple base models using various combination methods including weighted mean,
median, and stacking approaches.

The EnsemblePredictor properly implements the BasePredictor interface with
fit(X, y) and predict(X) methods, enabling seamless integration with the
SignalForge prediction pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import structlog
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.optimize import minimize

from signalforge.ml.models.base import BasePredictor, PredictionResult

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class EnsembleConfig(BaseModel):
    """Configuration for ensemble models."""

    combination_method: Literal["mean", "weighted_mean", "median", "stack"] = "weighted_mean"
    weights: list[float] | None = None
    optimize_weights: bool = False
    optimization_metric: Literal["mse", "mae"] = "mse"


@dataclass
class EnsemblePrediction:
    """Result of an ensemble prediction with model contributions."""

    prediction: float
    confidence: float
    lower_bound: float
    upper_bound: float
    model_predictions: dict[str, float] = field(default_factory=dict)
    weights_used: list[float] = field(default_factory=list)


class EnsemblePredictor(BasePredictor):
    """Ensemble model combining multiple predictors.

    This class combines predictions from multiple base models using various
    combination methods to improve prediction accuracy and robustness. It supports
    equal weighting, custom weights, weighted mean, median, and stacking approaches.

    The ensemble automatically optimizes weights when configured to do so, using
    cross-validation to find the optimal combination that minimizes prediction error.

    Attributes:
        model_name: Human-readable name of the model.
        model_version: Version string for the model.

    Examples:
        Basic usage with weighted mean:

        >>> import polars as pl
        >>> from signalforge.ml.models.ensemble import EnsemblePredictor
        >>> from signalforge.ml.models.baseline import ARIMAPredictor
        >>>
        >>> model1 = ARIMAPredictor(order=(1, 1, 1))
        >>> model2 = ARIMAPredictor(order=(2, 1, 1))
        >>> ensemble = EnsemblePredictor(
        ...     models=[model1, model2],
        ...     combination_method="weighted_mean"
        ... )
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)

        With custom weights:

        >>> ensemble = EnsemblePredictor(
        ...     models=[model1, model2],
        ...     weights=[0.7, 0.3],
        ...     combination_method="weighted_mean"
        ... )
    """

    model_name: str = "ensemble"
    model_version: str = "1.0.0"

    def __init__(
        self,
        models: Sequence[BasePredictor],
        weights: list[float] | None = None,
        combination_method: Literal["mean", "weighted_mean", "median", "stack"] = "weighted_mean",
    ) -> None:
        """Initialize the ensemble predictor.

        Args:
            models: Sequence of BasePredictor instances to combine.
            weights: Optional custom weights for each model. If None, weights are
                    equal or optimized based on combination_method. Must sum to 1.0
                    and be non-negative.
            combination_method: Method for combining predictions:
                - "mean": Simple arithmetic mean
                - "weighted_mean": Weighted average (uses weights parameter)
                - "median": Median of predictions (robust to outliers)
                - "stack": Use meta-learner (Ridge) to combine predictions

        Raises:
            ValueError: If models list is empty, weights don't match number of models,
                       or weights are invalid.
        """
        if not models:
            raise ValueError("At least one model must be provided")

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(models)})"
                )
            if any(w < 0 for w in weights):
                raise ValueError("All weights must be non-negative")
            total_weight = sum(weights)
            if not np.isclose(total_weight, 1.0):
                raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.models = list(models)
        self._weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        self.combination_method = combination_method
        self._fitted = False
        self._meta_model: Any = None  # For stacking
        self._feature_columns: list[str] = []

        logger.info(
            "ensemble_initialized",
            n_models=len(models),
            combination_method=combination_method,
            weights=self._weights,
        )

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> EnsemblePredictor:
        """Train all component models.

        Fits each base model on the provided data. If using stacking, also trains
        a meta-learner on the predictions of base models.

        Args:
            X: Feature matrix with historical data.
            y: Target series containing values to predict.
            **kwargs: Additional parameters:
                - optimize_weights: bool, whether to optimize weights (default: False)
                - metric: str, metric to optimize ("mse" or "mae", default: "mse")

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

        logger.info("fitting_ensemble", n_samples=X.height, n_features=len(X.columns))

        self._feature_columns = X.columns

        try:
            # Fit all base models
            for idx, model in enumerate(self.models):
                logger.debug(f"fitting_base_model_{idx}", model_type=type(model).__name__)
                model.fit(X, y)
                logger.debug(f"base_model_{idx}_fitted")

            # Optimize weights if requested and not using stacking
            if kwargs.get("optimize_weights", False) and self.combination_method != "stack":
                metric = kwargs.get("metric", "mse")
                logger.info("optimizing_weights", metric=metric)
                self._weights = self.optimize_weights(X, y, metric=metric)
                logger.info("weights_optimized", weights=self._weights)

            # Train meta-model if using stacking
            if self.combination_method == "stack":
                logger.info("training_meta_model")
                self._train_meta_model(X, y)
                logger.info("meta_model_trained")

            self._fitted = True
            logger.info("ensemble_fitted_successfully")

            return self

        except Exception as e:
            logger.error("ensemble_fitting_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to fit ensemble: {e}") from e

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Generate combined predictions.

        Combines predictions from all base models using the configured method.

        Args:
            X: Feature matrix for which to generate predictions.

        Returns:
            List of PredictionResult objects, one per row in X.

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

        logger.info("generating_ensemble_predictions", n_samples=X.height)

        try:
            # Get predictions from all base models
            all_predictions: list[list[PredictionResult]] = []
            for idx, model in enumerate(self.models):
                logger.debug(f"getting_predictions_from_model_{idx}")
                preds = model.predict(X)
                all_predictions.append(preds)

            # Combine predictions
            if self.combination_method == "stack" and self._meta_model is not None:
                results = self._predict_with_stacking(all_predictions)
            else:
                results = self._combine_predictions(all_predictions)

            logger.info("ensemble_predictions_generated", n_predictions=len(results))
            return results

        except Exception as e:
            logger.error("ensemble_prediction_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to generate predictions: {e}") from e

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return combined predictions with confidence from model agreement.

        Calculates confidence based on agreement between models. Higher agreement
        (lower standard deviation) indicates higher confidence.

        Args:
            X: Feature matrix for which to generate probability predictions.

        Returns:
            DataFrame with columns:
            - prediction: Combined prediction value
            - confidence: Confidence score based on model agreement (0.0-1.0)
            - std_dev: Standard deviation of base model predictions
            - model_0, model_1, ...: Individual model predictions

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has incompatible features.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        logger.info("generating_ensemble_proba", n_samples=X.height)

        # Get predictions from all models
        all_predictions: list[list[PredictionResult]] = []
        for model in self.models:
            preds = model.predict(X)
            all_predictions.append(preds)

        # Build DataFrame with all predictions
        rows: list[dict[str, Any]] = []

        for i in range(X.height):
            # Extract predictions for this sample from all models
            sample_preds = [
                all_predictions[model_idx][i].prediction
                for model_idx in range(len(self.models))
            ]

            pred_array = np.array(sample_preds)
            mean_pred = float(np.mean(pred_array))
            std_dev = float(np.std(pred_array))

            # Confidence based on agreement (inverse of coefficient of variation)
            # Higher agreement (lower std) means higher confidence
            if mean_pred != 0:
                cv = std_dev / abs(mean_pred)
                confidence = float(1.0 / (1.0 + cv))
            else:
                confidence = 0.5

            row: dict[str, Any] = {
                "prediction": mean_pred,
                "confidence": confidence,
                "std_dev": std_dev,
            }

            # Add individual model predictions
            for model_idx, pred_val in enumerate(sample_preds):
                row[f"model_{model_idx}"] = pred_val

            rows.append(row)

        result_df = pl.DataFrame(rows)
        logger.info("ensemble_proba_generated")
        return result_df

    def optimize_weights(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        metric: str = "mse",
    ) -> list[float]:
        """Optimize ensemble weights using validation data.

        Uses scipy.optimize to find optimal weights that minimize the specified
        metric on the validation data.

        Args:
            X: Validation feature matrix.
            y: Validation target series.
            metric: Metric to optimize ("mse" or "mae").

        Returns:
            List of optimized weights summing to 1.0.

        Raises:
            ValueError: If metric is unsupported or optimization fails.
            RuntimeError: If models are not fitted.
        """
        logger.info("optimizing_ensemble_weights", metric=metric, n_samples=X.height)

        # Check if models are fitted (assuming they have _fitted attribute)
        for model in self.models:
            if hasattr(model, "_fitted") and not model._fitted:
                raise RuntimeError("All models must be fitted before weight optimization")

        # Get predictions from all models
        all_preds: list[NDArray[np.float64]] = []
        for model in self.models:
            pred_results = model.predict(X)
            preds = np.array([p.prediction for p in pred_results])
            all_preds.append(preds)

        # Stack predictions into matrix (n_samples, n_models)
        pred_matrix = np.column_stack(all_preds)
        y_actual = y.to_numpy()

        # Define objective function
        def objective(weights: NDArray[np.float64]) -> float:
            weighted_pred = pred_matrix @ weights
            if metric == "mse":
                error = float(np.mean((weighted_pred - y_actual) ** 2))
            elif metric == "mae":
                error = float(np.mean(np.abs(weighted_pred - y_actual)))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            return error

        # Initial guess: equal weights
        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models

        # Constraints: weights sum to 1, all non-negative
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning("weight_optimization_failed", message=result.message)
            # Fall back to equal weights
            return [1.0 / n_models] * n_models

        optimized_weights: list[float] = result.x.tolist()
        logger.info("weights_optimized_successfully", weights=optimized_weights, error=result.fun)
        return optimized_weights

    def get_model_contributions(self, X: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Return individual model predictions for analysis.

        Args:
            X: Feature matrix for which to get predictions.

        Returns:
            Dictionary mapping model indices to their prediction DataFrames.
            Each DataFrame contains all PredictionResult fields.

        Raises:
            RuntimeError: If model not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting contributions")

        logger.info("getting_model_contributions", n_samples=X.height)

        contributions: dict[str, pl.DataFrame] = {}

        for idx, model in enumerate(self.models):
            pred_results = model.predict(X)

            # Convert PredictionResults to DataFrame
            rows = []
            for pred in pred_results:
                rows.append({
                    "symbol": pred.symbol,
                    "timestamp": pred.timestamp,
                    "horizon_days": pred.horizon_days,
                    "prediction": pred.prediction,
                    "confidence": pred.confidence,
                    "lower_bound": pred.lower_bound,
                    "upper_bound": pred.upper_bound,
                    "model_name": pred.model_name,
                    "model_version": pred.model_version,
                })

            contributions[f"model_{idx}"] = pl.DataFrame(rows)

        logger.info("model_contributions_retrieved", n_models=len(contributions))
        return contributions

    def _combine_predictions(
        self,
        all_predictions: list[list[PredictionResult]],
    ) -> list[PredictionResult]:
        """Combine predictions using configured method.

        Args:
            all_predictions: List of prediction lists, one per model.

        Returns:
            Combined predictions.
        """
        n_samples = len(all_predictions[0])
        combined: list[PredictionResult] = []

        for i in range(n_samples):
            # Extract predictions for this sample
            sample_preds = [
                all_predictions[model_idx][i].prediction
                for model_idx in range(len(self.models))
            ]

            pred_array = np.array(sample_preds)

            # Combine based on method
            if self.combination_method == "mean":
                final_pred = float(np.mean(pred_array))
            elif self.combination_method == "weighted_mean":
                final_pred = float(np.sum(pred_array * np.array(self._weights)))
            elif self.combination_method == "median":
                final_pred = float(np.median(pred_array))
            else:
                final_pred = float(np.mean(pred_array))

            # Calculate confidence based on agreement
            std_dev = float(np.std(pred_array))
            mean_val = float(np.mean(pred_array))
            if mean_val != 0:
                cv = std_dev / abs(mean_val)
                confidence = float(1.0 / (1.0 + cv))
            else:
                confidence = 0.5

            # Calculate bounds from spread
            lower_bound = final_pred - 1.96 * std_dev
            upper_bound = final_pred + 1.96 * std_dev

            # Use first model's metadata as template
            template = all_predictions[0][i]

            result = PredictionResult(
                symbol=template.symbol,
                timestamp=template.timestamp if template.timestamp else datetime.now(),
                horizon_days=template.horizon_days,
                prediction=final_pred,
                confidence=confidence,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            combined.append(result)

        return combined

    def _train_meta_model(self, X: pl.DataFrame, y: pl.Series) -> None:
        """Train meta-model for stacking.

        Args:
            X: Training features.
            y: Training targets.
        """
        from sklearn.linear_model import Ridge

        # Get predictions from all base models
        all_preds: list[NDArray[np.float64]] = []
        for model in self.models:
            pred_results = model.predict(X)
            preds = np.array([p.prediction for p in pred_results])
            all_preds.append(preds)

        # Stack into feature matrix for meta-model
        meta_X = np.column_stack(all_preds)
        meta_y = y.to_numpy()

        # Train Ridge meta-model
        self._meta_model = Ridge(alpha=1.0)
        self._meta_model.fit(meta_X, meta_y)

        logger.info(
            "meta_model_trained",
            coefficients=self._meta_model.coef_.tolist(),
            intercept=float(self._meta_model.intercept_),
        )

    def _predict_with_stacking(
        self,
        all_predictions: list[list[PredictionResult]],
    ) -> list[PredictionResult]:
        """Generate predictions using stacking meta-model.

        Args:
            all_predictions: Predictions from all base models.

        Returns:
            Combined predictions from meta-model.
        """
        if self._meta_model is None:
            raise RuntimeError("Meta-model not trained")

        n_samples = len(all_predictions[0])
        combined: list[PredictionResult] = []

        for i in range(n_samples):
            # Extract predictions for this sample
            sample_preds = [
                all_predictions[model_idx][i].prediction
                for model_idx in range(len(self.models))
            ]

            # Predict with meta-model
            meta_X = np.array([sample_preds])
            final_pred = float(self._meta_model.predict(meta_X)[0])

            # Calculate confidence from base model agreement
            pred_array = np.array(sample_preds)
            std_dev = float(np.std(pred_array))
            mean_val = float(np.mean(pred_array))
            if mean_val != 0:
                cv = std_dev / abs(mean_val)
                confidence = float(1.0 / (1.0 + cv))
            else:
                confidence = 0.5

            # Calculate bounds
            lower_bound = final_pred - 1.96 * std_dev
            upper_bound = final_pred + 1.96 * std_dev

            # Use first model's metadata
            template = all_predictions[0][i]

            result = PredictionResult(
                symbol=template.symbol,
                timestamp=template.timestamp if template.timestamp else datetime.now(),
                horizon_days=template.horizon_days,
                prediction=final_pred,
                confidence=confidence,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                model_name=f"{self.model_name}_stack",
                model_version=self.model_version,
            )
            combined.append(result)

        return combined


class WeightedEnsemble(EnsemblePredictor):
    """Weighted ensemble using weighted mean combination."""

    model_name: str = "weighted_ensemble"

    def __init__(
        self,
        models: Sequence[BasePredictor],
        weights: list[float] | None = None,
    ) -> None:
        """Initialize weighted ensemble."""
        super().__init__(models, weights, combination_method="weighted_mean")


class StackingEnsemble(EnsemblePredictor):
    """Stacking ensemble using a meta-learner."""

    model_name: str = "stacking_ensemble"

    def __init__(
        self,
        models: Sequence[BasePredictor],
    ) -> None:
        """Initialize stacking ensemble."""
        super().__init__(models, combination_method="stack")


def create_ensemble(
    models: Sequence[BasePredictor],
    config: EnsembleConfig | None = None,
) -> EnsemblePredictor:
    """Factory function to create an ensemble from configuration.

    Args:
        models: List of base predictors.
        config: Ensemble configuration.

    Returns:
        Configured ensemble predictor.
    """
    if config is None:
        config = EnsembleConfig()

    if config.combination_method == "stack":
        return StackingEnsemble(models)
    else:
        return WeightedEnsemble(models, weights=config.weights)


def optimize_weights(
    models: Sequence[BasePredictor],
    X: pl.DataFrame,
    y: pl.Series,
    metric: Literal["mse", "mae"] = "mse",
) -> list[float]:
    """Optimize ensemble weights for given models and data.

    Args:
        models: List of fitted base predictors.
        X: Validation feature matrix.
        y: Validation target series.
        metric: Metric to optimize.

    Returns:
        Optimized weights summing to 1.0.
    """
    ensemble = EnsemblePredictor(models, combination_method="weighted_mean")
    return ensemble.optimize_weights(X, y, metric=metric)
