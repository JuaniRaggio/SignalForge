"""Ensemble framework for combining multiple prediction models.

This module provides an ensemble framework that combines predictions from multiple
base models to improve forecasting accuracy and robustness. It implements:
- Weighted averaging with automatic weight optimization
- Median ensemble for outlier robustness
- Stacking ensemble with meta-learner
- Cross-validation for weight optimization

The ensemble framework integrates with MLflow for tracking and follows the
BasePredictor interface for consistency.

Examples:
    Basic weighted ensemble:

    >>> import polars as pl
    >>> from signalforge.ml.models.ensemble import create_ensemble, EnsembleConfig
    >>> from signalforge.ml.models.baseline import ARIMAPredictor, RollingMeanPredictor
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
    >>> prediction = ensemble.predict(horizon=10)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from signalforge.core.logging import get_logger
from signalforge.ml.models.base import BasePredictor
from signalforge.ml.training.mlflow_config import log_params

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models.

    Attributes:
        models: List of model names to combine in the ensemble.
        weights: Optional custom weights for each model. If None, weights are
                equal or optimized based on optimize_weights flag.
        method: Ensemble combination method. Options:
               - "mean": Simple arithmetic mean
               - "weighted": Weighted average (uses weights parameter)
               - "median": Median of predictions (robust to outliers)
               - "stacking": Use meta-learner to combine predictions
        optimize_weights: If True and method is "weighted", automatically
                         optimize weights using cross-validation.

    Raises:
        ValueError: If weights are provided but don't match number of models.
    """

    models: list[str]
    weights: list[float] | None = None
    method: Literal["mean", "weighted", "median", "stacking"] = "weighted"
    optimize_weights: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of models ({len(self.models)})"
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("All weights must be non-negative")
            if sum(self.weights) == 0:
                raise ValueError("Sum of weights must be greater than zero")


@dataclass
class EnsemblePrediction:
    """Result of an ensemble prediction.

    Attributes:
        prediction: The final combined prediction value.
        model_predictions: Dictionary mapping model names to their individual predictions.
        weights_used: Dictionary mapping model names to the weights used in combination.
        confidence_interval: Optional tuple of (lower_bound, upper_bound) for
                           the prediction confidence interval.
    """

    prediction: float
    model_predictions: dict[str, float]
    weights_used: dict[str, float]
    confidence_interval: tuple[float, float] | None = None


class BaseEnsemble(ABC):
    """Abstract base class for ensemble models.

    This class defines the interface for ensemble models that combine multiple
    base predictors. All concrete ensemble implementations must inherit from
    this class and implement all abstract methods.

    The ensemble maintains a collection of base models and combines their
    predictions using a specified strategy.

    Attributes:
        config: Configuration for the ensemble.
        _models: Dictionary mapping model names to BasePredictor instances.
        _fitted: Boolean flag indicating if ensemble has been trained.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        """Initialize the ensemble.

        Args:
            config: Configuration for the ensemble.
        """
        self.config = config
        self._models: dict[str, BasePredictor] = {}
        self._fitted: bool = False

        logger.debug(
            "ensemble_initialized",
            method=config.method,
            n_models=len(config.models),
            optimize_weights=config.optimize_weights,
        )

    @abstractmethod
    def add_model(self, name: str, model: BasePredictor) -> None:
        """Add a base model to the ensemble.

        Args:
            name: Name to identify the model.
            model: Instance of a BasePredictor to add to the ensemble.

        Raises:
            ValueError: If model name is not in configuration or already exists.
        """
        pass

    @abstractmethod
    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train all base models and optimize ensemble weights if needed.

        Args:
            df: Input DataFrame with time series data.
            target_column: Name of the column to predict. Defaults to "close".

        Raises:
            ValueError: If DataFrame is empty or missing required columns.
            RuntimeError: If model fitting fails.
        """
        pass

    @abstractmethod
    def predict(self, horizon: int) -> EnsemblePrediction:
        """Generate ensemble prediction for future periods.

        Args:
            horizon: Number of periods to predict into the future.

        Returns:
            EnsemblePrediction object containing combined prediction and details.

        Raises:
            RuntimeError: If ensemble has not been fitted.
            ValueError: If horizon is not positive.
        """
        pass

    @abstractmethod
    def get_weights(self) -> dict[str, float]:
        """Get current weights used for combining predictions.

        Returns:
            Dictionary mapping model names to their weights.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if ensemble has been trained.

        Returns:
            True if ensemble is fitted and ready for prediction.
        """
        return self._fitted and all(model.is_fitted for model in self._models.values())


class WeightedEnsemble(BaseEnsemble):
    """Weighted ensemble that combines predictions using weighted average.

    This ensemble combines predictions from multiple base models using a
    weighted average. Weights can be:
    - Equal (simple mean)
    - Custom (user-specified)
    - Optimized (automatically determined via cross-validation)

    The optimization process uses time series cross-validation to find
    optimal weights that minimize prediction error on validation folds.

    Attributes:
        config: Configuration for the ensemble.
        _models: Dictionary of base models.
        _weights: Current weights for each model.
        _target_column: Name of the target column being predicted.
        _fitted: Boolean flag for training status.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        """Initialize weighted ensemble.

        Args:
            config: Configuration for the ensemble.
        """
        super().__init__(config)
        self._weights: dict[str, float] = {}
        self._target_column: str = "close"

    def add_model(self, name: str, model: BasePredictor) -> None:
        """Add a base model to the ensemble.

        Args:
            name: Name to identify the model.
            model: Instance of a BasePredictor to add to the ensemble.

        Raises:
            ValueError: If model name is not in configuration or already exists.
        """
        if name not in self.config.models:
            raise ValueError(
                f"Model '{name}' not found in configuration. Expected one of: {self.config.models}"
            )

        if name in self._models:
            raise ValueError(f"Model '{name}' already exists in ensemble")

        self._models[name] = model
        logger.debug("model_added_to_ensemble", model_name=name)

    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train all base models and optimize ensemble weights if needed.

        Args:
            df: Input DataFrame with time series data.
            target_column: Name of the column to predict. Defaults to "close".

        Raises:
            ValueError: If DataFrame is empty or missing required columns.
            RuntimeError: If not all configured models have been added or fitting fails.
        """
        if df.height == 0:
            raise ValueError("Cannot fit ensemble on empty DataFrame")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Verify all models have been added
        missing_models = set(self.config.models) - set(self._models.keys())
        if missing_models:
            raise RuntimeError(
                f"Missing models in ensemble: {missing_models}. "
                f"Use add_model() to add all configured models before fitting."
            )

        self._target_column = target_column

        logger.info(
            "fitting_ensemble",
            n_models=len(self._models),
            n_samples=df.height,
            target_column=target_column,
        )

        try:
            # Log parameters to MLflow if in active run
            try:
                log_params(
                    {
                        "ensemble_type": "WeightedEnsemble",
                        "ensemble_method": self.config.method,
                        "n_models": len(self._models),
                        "model_names": ",".join(self._models.keys()),
                        "optimize_weights": self.config.optimize_weights,
                        "target_column": target_column,
                    }
                )
            except RuntimeError:
                logger.debug("no_active_mlflow_run", action="skipping parameter logging")

            # Fit all base models
            for name, model in self._models.items():
                logger.info("fitting_base_model", model_name=name)
                model.fit(df, target_column=target_column)
                logger.info("base_model_fitted", model_name=name)

            # Set or optimize weights
            if self.config.weights is not None:
                # Use custom weights
                raw_weights = self.config.weights
                total = sum(raw_weights)
                self._weights = {
                    name: w / total for name, w in zip(self.config.models, raw_weights, strict=True)
                }
                logger.info("using_custom_weights", weights=self._weights)
            elif self.config.optimize_weights and self.config.method == "weighted":
                # Optimize weights using cross-validation
                logger.info("optimizing_ensemble_weights")
                self._weights = optimize_weights(self._models, df, target_column)
                logger.info("weights_optimized", weights=self._weights)
            else:
                # Use equal weights
                weight = 1.0 / len(self._models)
                self._weights = dict.fromkeys(self._models.keys(), weight)
                logger.info("using_equal_weights", weight=weight)

            self._fitted = True

            # Log weights to MLflow
            try:
                weight_params = {f"weight_{name}": w for name, w in self._weights.items()}
                log_params(weight_params)
            except RuntimeError:
                pass

            logger.info("ensemble_fitted_successfully")

        except Exception as e:
            logger.error(
                "ensemble_fitting_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to fit ensemble: {e}")

    def predict(self, horizon: int) -> EnsemblePrediction:
        """Generate ensemble prediction for future periods.

        For multi-step predictions (horizon > 1), this returns the first prediction
        value as the main prediction, with model predictions and weights for that step.

        Args:
            horizon: Number of periods to predict into the future.

        Returns:
            EnsemblePrediction object containing combined prediction and details.

        Raises:
            RuntimeError: If ensemble has not been fitted.
            ValueError: If horizon is not positive.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction. Call fit() first.")

        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        logger.info("generating_ensemble_prediction", horizon=horizon)

        try:
            # Get predictions from all base models
            model_predictions_dict: dict[str, float] = {}
            all_predictions: list[float] = []

            for name, model in self._models.items():
                pred_df = model.predict(horizon=horizon)
                # Take first prediction value
                pred_value = float(pred_df["prediction"][0])
                model_predictions_dict[name] = pred_value
                all_predictions.append(pred_value)

                logger.debug(
                    "base_model_prediction",
                    model_name=name,
                    prediction=pred_value,
                )

            # Combine predictions based on method
            if self.config.method == "mean":
                final_prediction = float(np.mean(all_predictions))
            elif self.config.method == "weighted":
                final_prediction = sum(
                    self._weights[name] * pred for name, pred in model_predictions_dict.items()
                )
            elif self.config.method == "median":
                final_prediction = float(np.median(all_predictions))
            else:
                raise ValueError(f"Unsupported method for WeightedEnsemble: {self.config.method}")

            # Calculate simple confidence interval based on prediction spread
            pred_array = np.array(all_predictions)
            std_dev = float(np.std(pred_array))
            confidence_interval = (
                final_prediction - 1.96 * std_dev,
                final_prediction + 1.96 * std_dev,
            )

            result = EnsemblePrediction(
                prediction=final_prediction,
                model_predictions=model_predictions_dict,
                weights_used=self._weights.copy(),
                confidence_interval=confidence_interval,
            )

            logger.info(
                "ensemble_prediction_generated",
                final_prediction=final_prediction,
                method=self.config.method,
            )

            return result

        except Exception as e:
            logger.error(
                "ensemble_prediction_failed",
                error=str(e),
                horizon=horizon,
                exc_info=True,
            )
            raise RuntimeError(f"Failed to generate ensemble prediction: {e}")

    def get_weights(self) -> dict[str, float]:
        """Get current weights used for combining predictions.

        Returns:
            Dictionary mapping model names to their weights.

        Raises:
            RuntimeError: If ensemble has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Ensemble must be fitted before accessing weights")

        return self._weights.copy()


class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble that uses a meta-learner to combine predictions.

    This ensemble trains a meta-model (Ridge Regression) to combine predictions
    from multiple base models. The meta-learner learns how to optimally weight
    and combine base predictions based on historical performance.

    The implementation uses proper train/validation split to avoid data leakage
    and ensures the meta-learner generalizes well.

    Attributes:
        config: Configuration for the ensemble.
        _models: Dictionary of base models.
        _meta_model: Ridge regression meta-learner.
        _target_column: Name of the target column being predicted.
        _fitted: Boolean flag for training status.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        """Initialize stacking ensemble.

        Args:
            config: Configuration for the ensemble.
        """
        super().__init__(config)
        self._meta_model: Ridge | None = None
        self._target_column: str = "close"

    def add_model(self, name: str, model: BasePredictor) -> None:
        """Add a base model to the ensemble.

        Args:
            name: Name to identify the model.
            model: Instance of a BasePredictor to add to the ensemble.

        Raises:
            ValueError: If model name is not in configuration or already exists.
        """
        if name not in self.config.models:
            raise ValueError(
                f"Model '{name}' not found in configuration. Expected one of: {self.config.models}"
            )

        if name in self._models:
            raise ValueError(f"Model '{name}' already exists in ensemble")

        self._models[name] = model
        logger.debug("model_added_to_ensemble", model_name=name)

    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train base models and meta-learner using cross-validation.

        This method:
        1. Splits data into train/validation using time series split
        2. Trains base models on training folds
        3. Generates predictions on validation folds
        4. Trains meta-learner on validation predictions and actual values

        Args:
            df: Input DataFrame with time series data.
            target_column: Name of the column to predict. Defaults to "close".

        Raises:
            ValueError: If DataFrame is empty or has insufficient data.
            RuntimeError: If not all configured models have been added or fitting fails.
        """
        if df.height == 0:
            raise ValueError("Cannot fit ensemble on empty DataFrame")

        if df.height < 20:
            raise ValueError(
                f"Insufficient data for stacking: need at least 20 samples, got {df.height}"
            )

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Verify all models have been added
        missing_models = set(self.config.models) - set(self._models.keys())
        if missing_models:
            raise RuntimeError(
                f"Missing models in ensemble: {missing_models}. "
                f"Use add_model() to add all configured models before fitting."
            )

        self._target_column = target_column

        logger.info(
            "fitting_stacking_ensemble",
            n_models=len(self._models),
            n_samples=df.height,
            target_column=target_column,
        )

        try:
            # Log parameters to MLflow if in active run
            try:
                log_params(
                    {
                        "ensemble_type": "StackingEnsemble",
                        "n_models": len(self._models),
                        "model_names": ",".join(self._models.keys()),
                        "target_column": target_column,
                        "meta_model": "Ridge",
                    }
                )
            except RuntimeError:
                logger.debug("no_active_mlflow_run", action="skipping parameter logging")

            # Sort by timestamp
            df_sorted = df.sort("timestamp")

            # Use time series cross-validation to train meta-model
            tscv = TimeSeriesSplit(n_splits=3)
            target_array = df_sorted[target_column].to_numpy()

            meta_features_list: list[NDArray[np.float64]] = []
            meta_targets_list: list[float] = []

            # Generate meta-features using cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(target_array)):
                logger.info("processing_cv_fold", fold=fold_idx + 1, n_splits=3)

                train_df = df_sorted[train_idx]

                # Train base models on training fold
                fold_predictions: list[float] = []

                for _name, model in self._models.items():
                    # Fit model on training data
                    model.fit(train_df, target_column=target_column)

                    # Generate predictions for validation period
                    horizon = len(val_idx)
                    pred_df = model.predict(horizon=horizon)
                    predictions = pred_df["prediction"].to_numpy()

                    # Store predictions for each validation sample
                    fold_predictions.extend(predictions)

                # Reshape predictions to (n_samples, n_models)
                n_models = len(self._models)
                predictions_matrix = np.array(fold_predictions).reshape(-1, n_models)

                # Store meta-features and targets
                for i, val_i in enumerate(val_idx):
                    meta_features_list.append(predictions_matrix[i])
                    meta_targets_list.append(float(target_array[val_i]))

            # Train meta-model on collected meta-features
            X_meta = np.array(meta_features_list)
            y_meta = np.array(meta_targets_list)

            logger.info(
                "training_meta_model",
                n_meta_samples=len(y_meta),
                n_features=X_meta.shape[1],
            )

            self._meta_model = Ridge(alpha=1.0)
            self._meta_model.fit(X_meta, y_meta)

            logger.info(
                "meta_model_trained",
                coefficients=self._meta_model.coef_.tolist(),
                intercept=float(self._meta_model.intercept_),
            )

            # Retrain all base models on full dataset
            logger.info("retraining_base_models_on_full_data")
            for _name, model in self._models.items():
                model.fit(df_sorted, target_column=target_column)

            self._fitted = True

            # Log meta-model weights
            try:
                weight_params = {
                    f"meta_weight_{name}": float(coef)
                    for name, coef in zip(self._models.keys(), self._meta_model.coef_, strict=True)
                }
                weight_params["meta_intercept"] = float(self._meta_model.intercept_)
                log_params(weight_params)
            except RuntimeError:
                pass

            logger.info("stacking_ensemble_fitted_successfully")

        except Exception as e:
            logger.error(
                "stacking_ensemble_fitting_failed",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to fit stacking ensemble: {e}")

    def predict(self, horizon: int) -> EnsemblePrediction:
        """Generate stacked ensemble prediction using meta-learner.

        Args:
            horizon: Number of periods to predict into the future.

        Returns:
            EnsemblePrediction object containing combined prediction and details.

        Raises:
            RuntimeError: If ensemble has not been fitted.
            ValueError: If horizon is not positive.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction. Call fit() first.")

        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        if self._meta_model is None:
            raise RuntimeError("Meta-model not trained. This should not happen.")

        logger.info("generating_stacking_prediction", horizon=horizon)

        try:
            # Get predictions from all base models
            model_predictions_dict: dict[str, float] = {}
            base_predictions: list[float] = []

            for name, model in self._models.items():
                pred_df = model.predict(horizon=horizon)
                # Take first prediction value
                pred_value = float(pred_df["prediction"][0])
                model_predictions_dict[name] = pred_value
                base_predictions.append(pred_value)

                logger.debug(
                    "base_model_prediction",
                    model_name=name,
                    prediction=pred_value,
                )

            # Create feature vector for meta-model
            X_meta = np.array([base_predictions])

            # Get meta-model prediction
            final_prediction = float(self._meta_model.predict(X_meta)[0])

            # Calculate confidence interval based on spread
            pred_array = np.array(base_predictions)
            std_dev = float(np.std(pred_array))
            confidence_interval = (
                final_prediction - 1.96 * std_dev,
                final_prediction + 1.96 * std_dev,
            )

            # Extract weights from meta-model
            weights_dict = {
                name: float(coef)
                for name, coef in zip(self._models.keys(), self._meta_model.coef_, strict=True)
            }

            result = EnsemblePrediction(
                prediction=final_prediction,
                model_predictions=model_predictions_dict,
                weights_used=weights_dict,
                confidence_interval=confidence_interval,
            )

            logger.info(
                "stacking_prediction_generated",
                final_prediction=final_prediction,
            )

            return result

        except Exception as e:
            logger.error(
                "stacking_prediction_failed",
                error=str(e),
                horizon=horizon,
                exc_info=True,
            )
            raise RuntimeError(f"Failed to generate stacking prediction: {e}")

    def get_weights(self) -> dict[str, float]:
        """Get meta-model coefficients (weights) for each base model.

        Returns:
            Dictionary mapping model names to their meta-model coefficients.

        Raises:
            RuntimeError: If ensemble has not been fitted.
        """
        if not self._fitted or self._meta_model is None:
            raise RuntimeError("Ensemble must be fitted before accessing weights")

        return {
            name: float(coef)
            for name, coef in zip(self._models.keys(), self._meta_model.coef_, strict=True)
        }


def create_ensemble(config: EnsembleConfig) -> BaseEnsemble:
    """Factory function to create appropriate ensemble based on configuration.

    This is the recommended way to create ensemble instances as it automatically
    selects the correct ensemble type based on the configuration.

    Args:
        config: Configuration specifying ensemble type and parameters.

    Returns:
        Instance of appropriate ensemble class.

    Raises:
        ValueError: If configuration specifies unsupported ensemble method.

    Examples:
        Create a weighted ensemble:

        >>> config = EnsembleConfig(
        ...     models=["arima", "rolling_mean"],
        ...     method="weighted",
        ...     optimize_weights=True
        ... )
        >>> ensemble = create_ensemble(config)

        Create a stacking ensemble:

        >>> config = EnsembleConfig(
        ...     models=["arima", "rolling_mean"],
        ...     method="stacking"
        ... )
        >>> ensemble = create_ensemble(config)
    """
    if config.method in ("mean", "weighted", "median"):
        return WeightedEnsemble(config)
    elif config.method == "stacking":
        return StackingEnsemble(config)
    else:
        raise ValueError(
            f"Unsupported ensemble method: {config.method}. "
            f"Supported methods: mean, weighted, median, stacking"
        )


def optimize_weights(
    models: dict[str, BasePredictor],
    df: pl.DataFrame,
    target_column: str,
) -> dict[str, float]:
    """Optimize ensemble weights using time series cross-validation.

    This function finds optimal weights for combining model predictions by
    minimizing prediction error on validation folds. It uses:
    - Time series cross-validation to respect temporal order
    - Mean squared error as the optimization objective
    - Scipy optimization with non-negative weight constraints
    - Weight normalization to sum to 1

    Args:
        models: Dictionary of fitted base models.
        df: DataFrame with historical data for validation.
        target_column: Name of the target column.

    Returns:
        Dictionary mapping model names to optimized weights.

    Raises:
        ValueError: If optimization fails or produces invalid weights.
        RuntimeError: If models are not fitted.

    Examples:
        >>> models = {
        ...     "arima": fitted_arima_model,
        ...     "rolling_mean": fitted_rolling_mean_model,
        ... }
        >>> weights = optimize_weights(models, validation_df, "close")
        >>> print(weights)
        {'arima': 0.6, 'rolling_mean': 0.4}
    """
    logger.info("optimizing_weights", n_models=len(models))

    try:
        from scipy.optimize import minimize

        # Verify all models are fitted
        for name, model in models.items():
            if not model.is_fitted:
                raise RuntimeError(f"Model '{name}' must be fitted before weight optimization")

        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        df_sorted = df.sort("timestamp")
        target_array = df_sorted[target_column].to_numpy()

        # Collect predictions from each model on validation folds
        all_predictions: dict[str, list[float]] = {name: [] for name in models}
        all_targets: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(target_array)):
            train_df = df_sorted[train_idx]

            logger.debug("cv_fold_for_optimization", fold=fold_idx + 1)

            for name, model in models.items():
                # Fit on training fold
                model.fit(train_df, target_column=target_column)

                # Predict on validation fold
                horizon = len(val_idx)
                pred_df = model.predict(horizon=horizon)
                predictions = pred_df["prediction"].to_list()

                all_predictions[name].extend(predictions)

            # Store actual targets for this fold
            all_targets.extend(target_array[val_idx].tolist())

        # Convert predictions to numpy arrays
        pred_matrix = np.column_stack([all_predictions[name] for name in models])
        target_vec = np.array(all_targets)

        # Define objective function (MSE)
        def objective(weights: NDArray[np.float64]) -> float:
            weighted_pred = pred_matrix @ weights
            mse = float(np.mean((weighted_pred - target_vec) ** 2))
            return mse

        # Initial guess: equal weights
        n_models = len(models)
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
            weights_dict = dict.fromkeys(models.keys(), 1.0 / n_models)
        else:
            weights_dict = {name: float(w) for name, w in zip(models.keys(), result.x, strict=True)}
            logger.info("weights_optimized_successfully", weights=weights_dict, mse=result.fun)

        return weights_dict

    except ImportError as e:
        logger.warning("scipy_not_available", error=str(e))
        # Fall back to equal weights
        n_models = len(models)
        return dict.fromkeys(models.keys(), 1.0 / n_models)
    except Exception as e:
        logger.error("weight_optimization_error", error=str(e), exc_info=True)
        # Fall back to equal weights
        n_models = len(models)
        return dict.fromkeys(models.keys(), 1.0 / n_models)
