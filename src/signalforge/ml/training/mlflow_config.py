"""MLflow configuration and helper functions for experiment tracking.

This module provides utilities for:
- Setting up and managing MLflow experiments
- Logging parameters, metrics, and models
- Context manager for MLflow runs with proper cleanup
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

import mlflow
from mlflow.tracking import MlflowClient

from signalforge.core.config import get_settings
from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def _initialize_mlflow() -> None:
    """Initialize MLflow with configuration from settings.

    This function sets up the MLflow tracking URI from application settings.
    It is called automatically by other functions in this module.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.debug(
        "mlflow_initialized",
        tracking_uri=settings.mlflow_tracking_uri,
    )


def setup_experiment(experiment_name: str) -> str:
    """Setup or get existing MLflow experiment.

    Args:
        experiment_name: Name of the experiment to create or retrieve.

    Returns:
        The experiment ID as a string.

    Raises:
        Exception: If experiment creation or retrieval fails.
    """
    _initialize_mlflow()
    settings = get_settings()

    try:
        # Try to get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=settings.mlflow_artifact_location,
            )
            logger.info(
                "experiment_created",
                experiment_name=experiment_name,
                experiment_id=experiment_id,
            )
        else:
            experiment_id = experiment.experiment_id
            logger.debug(
                "experiment_exists",
                experiment_name=experiment_name,
                experiment_id=experiment_id,
            )

        return experiment_id

    except Exception as e:
        logger.error(
            "experiment_setup_failed",
            experiment_name=experiment_name,
            error=str(e),
            exc_info=True,
        )
        raise


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to the current MLflow run.

    Args:
        params: Dictionary of parameter names and values to log.
                Values will be converted to strings.

    Raises:
        RuntimeError: If no active run exists.
        Exception: If parameter logging fails.
    """
    if mlflow.active_run() is None:
        error_msg = "No active MLflow run. Use start_run() context manager first."
        logger.error("log_params_failed", error=error_msg)
        raise RuntimeError(error_msg)

    try:
        # Convert all values to strings as MLflow requires
        string_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(string_params)
        logger.debug(
            "params_logged",
            param_count=len(params),
            run_id=mlflow.active_run().info.run_id,
        )
    except Exception as e:
        logger.error(
            "log_params_failed",
            error=str(e),
            param_count=len(params),
            exc_info=True,
        )
        raise


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to the current MLflow run.

    Args:
        metrics: Dictionary of metric names and values to log.
        step: Optional step number for tracking metrics over time.

    Raises:
        RuntimeError: If no active run exists.
        Exception: If metric logging fails.
    """
    if mlflow.active_run() is None:
        error_msg = "No active MLflow run. Use start_run() context manager first."
        logger.error("log_metrics_failed", error=error_msg)
        raise RuntimeError(error_msg)

    try:
        mlflow.log_metrics(metrics, step=step)
        logger.debug(
            "metrics_logged",
            metric_count=len(metrics),
            step=step,
            run_id=mlflow.active_run().info.run_id,
        )
    except Exception as e:
        logger.error(
            "log_metrics_failed",
            error=str(e),
            metric_count=len(metrics),
            step=step,
            exc_info=True,
        )
        raise


def log_model(
    model: Any,
    artifact_path: str,
    registered_model_name: str | None = None,
) -> None:
    """Log a model artifact to MLflow.

    This function supports various ML frameworks including scikit-learn, PyTorch, TensorFlow, etc.
    MLflow will automatically detect the model type and use the appropriate flavor.

    Args:
        model: The trained model object to log.
        artifact_path: Relative path within the run's artifact directory to save the model.
        registered_model_name: If provided, register the model with this name in the MLflow Model Registry.

    Raises:
        RuntimeError: If no active run exists.
        Exception: If model logging fails.
    """
    if mlflow.active_run() is None:
        error_msg = "No active MLflow run. Use start_run() context manager first."
        logger.error("log_model_failed", error=error_msg)
        raise RuntimeError(error_msg)

    try:
        # Determine model flavor based on model type
        model_type = type(model).__name__
        model_module = type(model).__module__

        logger.debug(
            "logging_model",
            artifact_path=artifact_path,
            model_type=model_type,
            model_module=model_module,
            registered_model_name=registered_model_name,
        )

        # Use sklearn flavor for scikit-learn models
        if "sklearn" in model_module:
            import mlflow.sklearn

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        # Use pytorch flavor for PyTorch models
        elif "torch" in model_module:
            import mlflow.pytorch

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        # Use tensorflow flavor for TensorFlow/Keras models
        elif "tensorflow" in model_module or "keras" in model_module:
            import mlflow.tensorflow

            mlflow.tensorflow.log_model(
                model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        # Use xgboost flavor for XGBoost models
        elif "xgboost" in model_module:
            import mlflow.xgboost

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        # Use lightgbm flavor for LightGBM models
        elif "lightgbm" in model_module:
            import mlflow.lightgbm

            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        # Fallback to generic pyfunc flavor
        else:
            import mlflow.pyfunc

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=registered_model_name,
            )

        logger.info(
            "model_logged",
            artifact_path=artifact_path,
            model_type=model_type,
            registered_model_name=registered_model_name,
            run_id=mlflow.active_run().info.run_id,
        )

    except Exception as e:
        logger.error(
            "log_model_failed",
            error=str(e),
            artifact_path=artifact_path,
            model_type=type(model).__name__,
            exc_info=True,
        )
        raise


@contextmanager
def start_run(
    run_name: str | None = None,
    nested: bool = False,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager for MLflow runs with automatic cleanup.

    This context manager ensures proper setup and teardown of MLflow runs,
    including error handling and logging.

    Args:
        run_name: Optional name for the run. If not provided, MLflow generates one.
        nested: If True, creates a nested run under the current active run.

    Yields:
        The active MLflow run object.

    Raises:
        Exception: If run creation or cleanup fails.

    Example:
        with start_run(run_name="training_experiment") as run:
            log_params({"learning_rate": 0.01})
            # ... training code ...
            log_metrics({"accuracy": 0.95})
    """
    _initialize_mlflow()
    settings = get_settings()

    # Get or create experiment
    experiment_id = setup_experiment(settings.mlflow_experiment_name)

    try:
        # Start the run
        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
            nested=nested,
        )

        logger.info(
            "run_started",
            run_id=run.info.run_id,
            run_name=run_name,
            experiment_id=experiment_id,
            nested=nested,
        )

        yield run

        logger.info(
            "run_completed",
            run_id=run.info.run_id,
            run_name=run_name,
        )

    except Exception as e:
        logger.error(
            "run_failed",
            run_name=run_name,
            error=str(e),
            exc_info=True,
        )
        raise

    finally:
        # Ensure run is ended even if an exception occurred
        if mlflow.active_run() is not None:
            mlflow.end_run()
            logger.debug("run_ended", run_name=run_name)
