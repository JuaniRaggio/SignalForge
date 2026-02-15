"""Training pipeline orchestrator for SignalForge ML models.

This module provides a comprehensive training pipeline that integrates:
- Feature engineering with technical indicators
- Walk-forward validation for time series
- MLflow experiment tracking
- Model training and evaluation
- Multi-model comparison

The TrainingPipeline class handles the complete workflow from raw OHLCV data
to trained models with validation metrics and experiment tracking.

Examples:
    Basic usage:

    >>> import polars as pl
    >>> from signalforge.ml.features.technical import TechnicalFeatureEngine
    >>> from signalforge.ml.training.validation import WalkForwardValidator
    >>> from signalforge.ml.training.pipeline import TrainingPipeline
    >>> from signalforge.ml.models.lstm import LSTMPredictor
    >>>
    >>> df = pl.DataFrame({...})  # OHLCV data
    >>> engine = TechnicalFeatureEngine()
    >>> validator = WalkForwardValidator(n_splits=5)
    >>> pipeline = TrainingPipeline(engine, validator)
    >>>
    >>> model = LSTMPredictor(input_size=10)
    >>> result = pipeline.train(model, df, target_column="close", horizon=5)

    With MLflow tracking:

    >>> from signalforge.ml.training.mlflow_utils import MLflowTracker
    >>> tracker = MLflowTracker(experiment_name="price_prediction")
    >>> pipeline = TrainingPipeline(engine, validator, mlflow_tracker=tracker)
    >>> result = pipeline.train(model, df)

    Training multiple models:

    >>> models = [
    ...     LSTMPredictor(input_size=10, hidden_size=64),
    ...     LSTMPredictor(input_size=10, hidden_size=128),
    ... ]
    >>> results = pipeline.train_multiple(models, df)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import polars as pl

    from signalforge.ml.features.technical import TechnicalFeatureEngine
    from signalforge.ml.models.base import BasePredictor
    from signalforge.ml.training.mlflow_utils import MLflowTracker
    from signalforge.ml.training.validation import ValidationResult, WalkForwardValidator

logger = structlog.get_logger(__name__)


@dataclass
class TrainingResult:
    """Results from model training pipeline.

    Attributes:
        model: Trained model instance.
        validation_result: Results from walk-forward validation (None if skipped).
        training_time: Time in seconds to train the model.
        final_metrics: Dictionary of final model metrics on full dataset.
        feature_importance: Dictionary mapping feature names to importance scores.

    Examples:
        >>> result = TrainingResult(
        ...     model=trained_model,
        ...     validation_result=val_result,
        ...     training_time=120.5,
        ...     final_metrics={"mse": 0.05, "mae": 0.15},
        ...     feature_importance={"sma_20": 0.3, "rsi_14": 0.2}
        ... )
    """

    model: BasePredictor
    validation_result: ValidationResult | None
    training_time: float
    final_metrics: dict[str, float]
    feature_importance: dict[str, float]


class TrainingPipeline:
    """Orchestrator for model training with feature engineering.

    This class provides a high-level interface for training time series models.
    It handles the complete pipeline including:
    1. Feature engineering from OHLCV data
    2. Target variable creation
    3. Walk-forward validation
    4. Final model training
    5. MLflow experiment tracking

    Attributes:
        feature_engine: Engine for computing technical indicators.
        validator: Walk-forward validator for time series cross-validation.
        mlflow_tracker: Optional MLflow tracker for experiment logging.

    Examples:
        >>> from signalforge.ml.features.technical import TechnicalFeatureEngine
        >>> from signalforge.ml.training.validation import WalkForwardValidator
        >>>
        >>> engine = TechnicalFeatureEngine()
        >>> validator = WalkForwardValidator(n_splits=5)
        >>> pipeline = TrainingPipeline(engine, validator)
        >>>
        >>> # Train a model
        >>> result = pipeline.train(model, df, target_column="close")
    """

    def __init__(
        self,
        feature_engine: TechnicalFeatureEngine,
        validator: WalkForwardValidator,
        mlflow_tracker: MLflowTracker | None = None,
    ):
        """Initialize the training pipeline.

        Args:
            feature_engine: Configured TechnicalFeatureEngine for computing features.
            validator: Configured WalkForwardValidator for cross-validation.
            mlflow_tracker: Optional MLflowTracker for experiment logging.

        Examples:
            >>> from signalforge.ml.features.technical import TechnicalFeatureEngine, FeatureConfig
            >>> from signalforge.ml.training.validation import WalkForwardValidator
            >>>
            >>> config = FeatureConfig(sma_periods=[5, 10, 20])
            >>> engine = TechnicalFeatureEngine(config)
            >>> validator = WalkForwardValidator(n_splits=5, test_size=20)
            >>> pipeline = TrainingPipeline(engine, validator)
        """
        self.feature_engine = feature_engine
        self.validator = validator
        self.mlflow_tracker = mlflow_tracker

        logger.info(
            "Initialized TrainingPipeline",
            has_mlflow=mlflow_tracker is not None,
            validator_splits=validator.n_splits,
        )

    def prepare_data(
        self,
        df: pl.DataFrame,
        target_column: str = "close",
        horizon: int = 5,
    ) -> tuple[pl.DataFrame, pl.Series]:
        """Prepare data for training.

        Performs the following steps:
        1. Computes technical features using feature engine
        2. Creates target variable (future returns)
        3. Drops NaN rows from feature computation and target creation

        Args:
            df: Input DataFrame with OHLCV data. Must contain:
                - timestamp: datetime
                - open, high, low, close: float
                - volume: float
            target_column: Column name to use for target prediction (default: "close").
            horizon: Number of periods ahead to predict (default: 5).

        Returns:
            Tuple of (features_df, target_series).

        Raises:
            ValueError: If required columns are missing or data is insufficient.

        Examples:
            >>> X, y = pipeline.prepare_data(df, target_column="close", horizon=5)
            >>> print(f"Features: {X.width} columns, {len(X)} rows")
            >>> print(f"Target: {len(y)} values")
        """
        import polars as pl

        # Validate input
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        if len(df) < horizon + 50:  # Minimum for feature computation
            raise ValueError(
                f"Insufficient data: need at least {horizon + 50} rows, got {len(df)}"
            )

        logger.info(
            "Preparing data",
            rows=len(df),
            target_column=target_column,
            horizon=horizon,
        )

        # Compute technical features
        df_with_features = self.feature_engine.compute_all(df)

        logger.info(
            "Technical features computed",
            total_columns=len(df_with_features.columns),
        )

        # Create target: future returns
        # Target is the percentage change horizon periods ahead
        df_with_target = df_with_features.with_columns(
            (
                (pl.col(target_column).shift(-horizon) - pl.col(target_column))
                / pl.col(target_column)
                * 100.0
            ).alias("target")
        )

        # Drop NaN values (any row with any NaN)
        # For training, we need complete cases only
        df_clean = df_with_target.filter(
            pl.all_horizontal(pl.all().is_not_null())
        )

        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after dropping NaN values")

        logger.info(
            "Data prepared",
            clean_rows=len(df_clean),
            dropped_rows=len(df_with_target) - len(df_clean),
        )

        # Separate features and target
        # Exclude original OHLCV columns and target from features
        feature_cols = [
            col
            for col in df_clean.columns
            if col
            not in ["timestamp", "open", "high", "low", "close", "volume", "target"]
        ]

        X = df_clean.select(["timestamp"] + feature_cols)
        y = df_clean.select("target").to_series()

        logger.info(
            "Features and target separated",
            n_features=len(feature_cols),
            n_samples=len(X),
        )

        return X, y

    def train(
        self,
        model: BasePredictor,
        df: pl.DataFrame,
        target_column: str = "close",
        horizon: int = 5,
        run_validation: bool = True,
    ) -> TrainingResult:
        """Full training pipeline.

        Executes the complete training workflow:
        1. Prepares data (features + target)
        2. Runs walk-forward validation (optional)
        3. Trains final model on all data
        4. Logs to MLflow (if tracker provided)

        Args:
            model: Model instance implementing BasePredictor interface.
            df: Input DataFrame with OHLCV data.
            target_column: Column to predict (default: "close").
            horizon: Prediction horizon in periods (default: 5).
            run_validation: Whether to run cross-validation (default: True).

        Returns:
            TrainingResult containing trained model and metrics.

        Raises:
            ValueError: If data is invalid or insufficient.
            RuntimeError: If training fails.

        Examples:
            >>> from signalforge.ml.models.lstm import LSTMPredictor
            >>> model = LSTMPredictor(input_size=50)
            >>> result = pipeline.train(model, df, horizon=5)
            >>> print(f"Training time: {result.training_time:.2f}s")
            >>> print(f"Validation MSE: {result.validation_result.mean_metrics['mse']:.4f}")
        """
        logger.info(
            "Starting training pipeline",
            model_name=model.model_name,
            target_column=target_column,
            horizon=horizon,
            run_validation=run_validation,
        )

        start_time = time.time()

        # Start MLflow run if tracker is available
        if self.mlflow_tracker is not None:
            run_name = f"{model.model_name}_{target_column}_h{horizon}"
            self.mlflow_tracker.start_run(
                run_name=run_name,
                tags={
                    "model_name": model.model_name,
                    "target_column": target_column,
                    "horizon": str(horizon),
                },
            )

        try:
            # Prepare data
            X, y = self.prepare_data(df, target_column=target_column, horizon=horizon)

            # Log parameters to MLflow
            if self.mlflow_tracker is not None:
                params = {
                    "target_column": target_column,
                    "horizon": horizon,
                    "n_samples": len(X),
                    "n_features": X.width - 1,  # Exclude timestamp
                }
                # Add model-specific parameters
                if hasattr(model, "__dict__"):
                    model_params = {
                        k: v
                        for k, v in model.__dict__.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                    params.update(model_params)

                self.mlflow_tracker.log_params(params)

            # Run validation if requested
            validation_result = None
            if run_validation:
                logger.info("Running walk-forward validation")
                # Include timestamp for the validator, but it will be dropped before model training
                # The validator needs timestamp for walk-forward splitting
                validation_result = self.validator.cross_validate(
                    model=model,
                    X=X,
                    y=y,
                    metrics=["mse", "mae", "rmse", "directional_accuracy"],
                )

                logger.info(
                    "Validation complete",
                    mean_mse=validation_result.mean_metrics.get("mse", 0.0),
                    mean_mae=validation_result.mean_metrics.get("mae", 0.0),
                )

                # Log validation results to MLflow
                if self.mlflow_tracker is not None:
                    self.mlflow_tracker.log_validation_results(validation_result)

            # Train final model on all data
            logger.info("Training final model on full dataset")
            X_train = X.drop("timestamp")  # Remove timestamp for training
            model.fit(X_train, y)

            # Calculate final metrics
            predictions = model.predict(X_train)
            y_pred = [p.prediction for p in predictions]

            import numpy as np

            y_true = y.to_numpy()
            # Align predictions with targets (may differ in length for LSTM)
            min_len = min(len(y_true), len(y_pred))
            y_true_aligned = y_true[-min_len:]
            y_pred_aligned = y_pred[-min_len:]

            final_metrics = {
                "mse": float(np.mean((y_true_aligned - y_pred_aligned) ** 2)),
                "mae": float(np.mean(np.abs(y_true_aligned - y_pred_aligned))),
                "rmse": float(np.sqrt(np.mean((y_true_aligned - y_pred_aligned) ** 2))),
            }

            logger.info("Final model trained", final_metrics=final_metrics)

            # Log final metrics to MLflow
            if self.mlflow_tracker is not None:
                final_metrics_prefixed = {
                    f"final_{k}": v for k, v in final_metrics.items()
                }
                self.mlflow_tracker.log_metrics(final_metrics_prefixed)

                # Log model
                try:
                    self.mlflow_tracker.log_model(model)
                except NotImplementedError:
                    logger.warning("Model does not support saving, skipping model logging")

            # Get feature importance
            feature_importance = model.get_feature_importance()

            # Calculate training time
            training_time = time.time() - start_time

            # Log training time to MLflow
            if self.mlflow_tracker is not None:
                self.mlflow_tracker.log_metrics({"training_time_seconds": training_time})

            logger.info(
                "Training pipeline complete",
                training_time=training_time,
                model_name=model.model_name,
            )

            result = TrainingResult(
                model=model,
                validation_result=validation_result,
                training_time=training_time,
                final_metrics=final_metrics,
                feature_importance=feature_importance,
            )

            return result

        except Exception as e:
            logger.error("Training pipeline failed", error=str(e))
            raise

        finally:
            # End MLflow run
            if self.mlflow_tracker is not None:
                self.mlflow_tracker.end_run()

    def train_multiple(
        self,
        models: list[BasePredictor],
        df: pl.DataFrame,
        **kwargs: Any,
    ) -> list[TrainingResult]:
        """Train multiple models and compare.

        Trains a list of models using the same data and hyperparameters,
        allowing for easy comparison. Each model is trained independently
        with its own MLflow run.

        Args:
            models: List of model instances to train.
            df: Input DataFrame with OHLCV data.
            **kwargs: Additional arguments passed to train() method.

        Returns:
            List of TrainingResult objects, one per model.

        Raises:
            ValueError: If models list is empty or data is invalid.
            RuntimeError: If any training fails.

        Examples:
            >>> models = [
            ...     LSTMPredictor(input_size=50, hidden_size=64),
            ...     LSTMPredictor(input_size=50, hidden_size=128),
            ... ]
            >>> results = pipeline.train_multiple(models, df, horizon=5)
            >>> for result in results:
            ...     print(f"{result.model.model_name}: MSE={result.final_metrics['mse']:.4f}")
        """
        if not models:
            raise ValueError("models list cannot be empty")

        logger.info("Training multiple models", n_models=len(models))

        results = []
        for i, model in enumerate(models):
            logger.info(
                f"Training model {i + 1}/{len(models)}",
                model_name=model.model_name,
            )

            try:
                result = self.train(model=model, df=df, **kwargs)
                results.append(result)

                logger.info(
                    f"Model {i + 1}/{len(models)} complete",
                    model_name=model.model_name,
                    training_time=result.training_time,
                )

            except Exception as e:
                logger.error(
                    f"Model {i + 1}/{len(models)} failed",
                    model_name=model.model_name,
                    error=str(e),
                )
                raise RuntimeError(
                    f"Training failed for model {model.model_name}: {e}"
                ) from e

        # Log comparison summary
        logger.info(
            "Multiple model training complete",
            n_models=len(models),
            total_time=sum(r.training_time for r in results),
        )

        # Print comparison
        logger.info("Model comparison:")
        for result in results:
            logger.info(
                "Model metrics",
                model_name=result.model.model_name,
                mse=result.final_metrics.get("mse", 0.0),
                mae=result.final_metrics.get("mae", 0.0),
                training_time=result.training_time,
            )

        return results


__all__ = ["TrainingPipeline", "TrainingResult"]
