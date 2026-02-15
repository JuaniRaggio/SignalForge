"""MLflow experiment tracking utilities for SignalForge.

This module provides a convenient wrapper around MLflow for tracking machine
learning experiments, including parameters, metrics, models, and artifacts.

The MLflowTracker class simplifies experiment tracking by providing a clean
interface for common operations and handling MLflow configuration.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from signalforge.ml.models.base import BasePredictor
    from signalforge.ml.training.validation import ValidationResult

logger = structlog.get_logger(__name__)


class MLflowTracker:
    """Wrapper for MLflow experiment tracking.

    This class provides a simplified interface for tracking machine learning
    experiments with MLflow. It handles experiment setup, run management,
    and logging of parameters, metrics, models, and artifacts.

    Attributes:
        experiment_name: Name of the MLflow experiment.
        experiment_id: MLflow experiment ID (set after initialization).
        active_run: Currently active MLflow run (None if no run is active).

    Examples:
        Basic usage:

        >>> from signalforge.ml.training.mlflow_utils import MLflowTracker
        >>>
        >>> tracker = MLflowTracker(experiment_name="price_prediction")
        >>> tracker.start_run(run_name="lstm_model_v1")
        >>> tracker.log_params({"learning_rate": 0.001, "epochs": 100})
        >>> tracker.log_metrics({"loss": 0.05, "accuracy": 0.95})
        >>> # tracker.log_model(model)
        >>> tracker.end_run()

        Using context manager:

        >>> tracker = MLflowTracker(experiment_name="price_prediction")
        >>> with tracker.run_context(run_name="lstm_model_v1"):
        ...     tracker.log_params({"learning_rate": 0.001})
        ...     tracker.log_metrics({"loss": 0.05})
        ...     # Training and evaluation code here
        ...     # Run automatically ends when context exits
    """

    def __init__(self, experiment_name: str = "signalforge") -> None:
        """Initialize the MLflow tracker.

        Creates or retrieves an MLflow experiment with the specified name.
        Sets up the tracking URI if configured in environment variables.

        Args:
            experiment_name: Name of the MLflow experiment. If the experiment
                           does not exist, it will be created.

        Examples:
            >>> tracker = MLflowTracker(experiment_name="my_experiment")
            >>> # Tracker is now ready to log runs
        """
        import mlflow

        self.experiment_name = experiment_name
        self.active_run: Any = None

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info("Created new MLflow experiment", experiment_name=experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(
                "Using existing MLflow experiment",
                experiment_name=experiment_name,
                experiment_id=self.experiment_id,
            )

    def start_run(
        self, run_name: str | None = None, tags: dict[str, str] | None = None
    ) -> None:
        """Start a new MLflow run.

        Creates a new run within the experiment for tracking a training session.
        If a run is already active, logs a warning and ends it before starting new one.

        Args:
            run_name: Optional name for the run. If None, MLflow generates a name.
            tags: Optional dictionary of tags to attach to the run.
                  Tags are useful for categorizing and filtering runs.

        Raises:
            RuntimeError: If MLflow fails to start the run.

        Examples:
            >>> tracker = MLflowTracker()
            >>> tracker.start_run(
            ...     run_name="baseline_model",
            ...     tags={"model_type": "arima", "version": "1.0"}
            ... )
        """
        import mlflow

        if self.active_run is not None:
            logger.warning("Run already active, ending it before starting new run")
            self.end_run()

        try:
            self.active_run = mlflow.start_run(
                experiment_id=self.experiment_id, run_name=run_name, tags=tags
            )
            logger.info(
                "Started MLflow run",
                run_name=run_name,
                run_id=self.active_run.info.run_id,
            )
        except Exception as e:
            logger.error("Failed to start MLflow run", error=str(e))
            raise RuntimeError(f"Failed to start MLflow run: {e}") from e

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the active run.

        Parameters are input settings that don't change during training
        (e.g., learning rate, model architecture settings).

        Args:
            params: Dictionary of parameter names and values.
                   Values are converted to strings for logging.

        Raises:
            RuntimeError: If no run is active or logging fails.

        Examples:
            >>> tracker.start_run()
            >>> tracker.log_params({
            ...     "learning_rate": 0.001,
            ...     "batch_size": 32,
            ...     "optimizer": "adam",
            ...     "hidden_layers": [128, 64, 32]
            ... })
        """
        import mlflow

        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        try:
            # MLflow requires string values for params
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
            logger.debug("Logged parameters", param_count=len(params))
        except Exception as e:
            logger.error("Failed to log parameters", error=str(e))
            raise RuntimeError(f"Failed to log parameters: {e}") from e

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log metrics to the active run.

        Metrics are numerical values that change during training
        (e.g., loss, accuracy, validation error).

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for time-series metrics.
                  Useful for tracking metrics across epochs or iterations.

        Raises:
            RuntimeError: If no run is active or logging fails.

        Examples:
            >>> tracker.start_run()
            >>> # Log epoch metrics
            >>> for epoch in range(10):
            ...     tracker.log_metrics(
            ...         {"train_loss": 0.5, "val_loss": 0.6},
            ...         step=epoch
            ...     )
            >>>
            >>> # Log final metrics
            >>> tracker.log_metrics({"final_accuracy": 0.95})
        """
        import mlflow

        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug("Logged metrics", metric_count=len(metrics), step=step)
        except Exception as e:
            logger.error("Failed to log metrics", error=str(e))
            raise RuntimeError(f"Failed to log metrics: {e}") from e

    def log_model(
        self, model: BasePredictor, artifact_path: str = "model"
    ) -> None:
        """Log a trained model to the active run.

        Saves the model as an artifact in MLflow, allowing it to be
        loaded later for inference or comparison.

        Args:
            model: Trained model instance implementing BasePredictor.
            artifact_path: Path within the run's artifact directory where
                          the model will be saved.

        Raises:
            RuntimeError: If no run is active or model logging fails.
            NotImplementedError: If the model doesn't support saving.

        Examples:
            >>> tracker.start_run()
            >>> # Train model
            >>> # model.fit(X_train, y_train)
            >>> tracker.log_model(model, artifact_path="trained_model")
        """
        import os
        import tempfile

        import mlflow

        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        try:
            # Save model to temporary file, then log to MLflow
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pkl")
                model.save(model_path)
                mlflow.log_artifact(model_path, artifact_path=artifact_path)

            logger.info(
                "Logged model",
                model_name=model.model_name,
                model_version=model.model_version,
                artifact_path=artifact_path,
            )
        except NotImplementedError as e:
            logger.error("Model does not support saving", error=str(e))
            raise
        except Exception as e:
            logger.error("Failed to log model", error=str(e))
            raise RuntimeError(f"Failed to log model: {e}") from e

    def log_validation_results(self, results: ValidationResult) -> None:
        """Log walk-forward validation results to the active run.

        Logs cross-validation metrics (mean and std) and optionally logs
        the predictions DataFrame as an artifact.

        Args:
            results: ValidationResult from WalkForwardValidator.cross_validate().

        Raises:
            RuntimeError: If no run is active or logging fails.

        Examples:
            >>> from signalforge.ml.training.validation import WalkForwardValidator
            >>> tracker.start_run(run_name="cv_evaluation")
            >>> validator = WalkForwardValidator(n_splits=5)
            >>> # results = validator.cross_validate(model, X, y)
            >>> # tracker.log_validation_results(results)
        """
        import os
        import tempfile

        import mlflow

        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        try:
            # Log mean metrics with "cv_mean_" prefix
            mean_metrics = {f"cv_mean_{k}": v for k, v in results.mean_metrics.items()}
            self.log_metrics(mean_metrics)

            # Log std metrics with "cv_std_" prefix
            std_metrics = {f"cv_std_{k}": v for k, v in results.std_metrics.items()}
            self.log_metrics(std_metrics)

            # Log fold count
            n_folds = len(results.fold_info)
            mlflow.log_param("cv_n_folds", n_folds)

            # Log predictions as CSV artifact
            if not results.predictions.is_empty():
                with tempfile.TemporaryDirectory() as tmpdir:
                    pred_path = os.path.join(tmpdir, "cv_predictions.csv")
                    results.predictions.write_csv(pred_path)
                    mlflow.log_artifact(pred_path, artifact_path="validation")

            # Log fold info as JSON artifact
            import json

            with tempfile.TemporaryDirectory() as tmpdir:
                info_path = os.path.join(tmpdir, "fold_info.json")
                with open(info_path, "w") as f:
                    json.dump(results.fold_info, f, indent=2)
                mlflow.log_artifact(info_path, artifact_path="validation")

            logger.info(
                "Logged validation results",
                n_folds=n_folds,
                mean_metrics=results.mean_metrics,
            )
        except Exception as e:
            logger.error("Failed to log validation results", error=str(e))
            raise RuntimeError(f"Failed to log validation results: {e}") from e

    def end_run(self) -> None:
        """End the active MLflow run.

        Marks the current run as finished. Should be called after all
        logging for the run is complete.

        Examples:
            >>> tracker.start_run()
            >>> # ... training and logging ...
            >>> tracker.end_run()
        """
        import mlflow

        if self.active_run is not None:
            try:
                mlflow.end_run()
                logger.info("Ended MLflow run", run_id=self.active_run.info.run_id)
                self.active_run = None
            except Exception as e:
                logger.error("Failed to end MLflow run", error=str(e))
                # Still clear active_run to prevent further issues
                self.active_run = None

    @contextmanager
    def run_context(
        self, run_name: str | None = None, tags: dict[str, str] | None = None
    ) -> Iterator[None]:
        """Context manager for MLflow runs.

        Automatically starts a run when entering the context and ends it
        when exiting. Ensures the run is properly closed even if an
        exception occurs.

        Args:
            run_name: Optional name for the run.
            tags: Optional dictionary of tags for the run.

        Yields:
            None

        Examples:
            >>> tracker = MLflowTracker()
            >>> with tracker.run_context(run_name="experiment_1"):
            ...     tracker.log_params({"param1": 1})
            ...     tracker.log_metrics({"metric1": 0.5})
            ...     # Run automatically ends when exiting context
            >>>
            >>> # Even if an exception occurs, run is properly ended
            >>> try:
            ...     with tracker.run_context(run_name="experiment_2"):
            ...         tracker.log_params({"param1": 1})
            ...         raise ValueError("Something went wrong")
            ... except ValueError:
            ...     pass  # Run is still properly ended
        """
        try:
            self.start_run(run_name=run_name, tags=tags)
            yield
        finally:
            self.end_run()
