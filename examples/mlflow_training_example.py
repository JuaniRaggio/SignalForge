"""Example demonstrating MLflow experiment tracking in SignalForge.

This example shows how to:
1. Setup MLflow experiment
2. Log parameters and metrics
3. Track model training over multiple epochs
4. Save and register models

Run this example:
    python examples/mlflow_training_example.py

View results in MLflow UI:
    http://localhost:5000
"""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from signalforge.core.logging import configure_logging, get_logger
from signalforge.ml.training import log_metrics, log_model, log_params, start_run

# Configure logging
configure_logging(json_logs=False, log_level="INFO")
logger = get_logger(__name__)


def generate_sample_data(n_samples: int = 1000, random_state: int = 42):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }


def train_single_model():
    """Example 1: Train a single model with MLflow tracking."""
    logger.info("Example 1: Training single model with MLflow tracking")

    # Generate data
    X_train, X_test, y_train, y_test = generate_sample_data()

    # Start MLflow run
    with start_run(run_name="random_forest_baseline") as run:
        logger.info("Started MLflow run", run_id=run.info.run_id)

        # Define hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }

        # Log parameters
        log_params(params)
        logger.info("Logged parameters", params=params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        log_metrics(metrics)
        logger.info("Logged metrics", metrics=metrics)

        # Log model
        log_model(
            model=model,
            artifact_path="model",
            registered_model_name="random_forest_baseline",
        )
        logger.info("Model logged to MLflow")

        logger.info("Run completed", run_id=run.info.run_id)


def hyperparameter_tuning():
    """Example 2: Hyperparameter tuning with multiple runs."""
    logger.info("Example 2: Hyperparameter tuning")

    # Generate data
    X_train, X_test, y_train, y_test = generate_sample_data()

    # Different hyperparameter configurations
    configurations = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 300, "max_depth": 20},
    ]

    best_f1 = 0.0
    best_config = None

    # Parent run for the hyperparameter tuning experiment
    with start_run(run_name="hyperparameter_tuning_experiment") as parent_run:
        log_params({"experiment_type": "grid_search", "n_configs": len(configurations)})

        for idx, config in enumerate(configurations):
            # Nested run for each configuration
            run_name = f"config_{idx}_trees{config['n_estimators']}_depth{config['max_depth']}"

            with start_run(run_name=run_name, nested=True) as child_run:
                logger.info(
                    "Testing configuration",
                    config_id=idx,
                    config=config,
                    run_id=child_run.info.run_id,
                )

                # Add common parameters
                params = {
                    **config,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                }

                log_params(params)

                # Train model
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                # Evaluate
                metrics = evaluate_model(model, X_test, y_test)
                log_metrics(metrics)

                # Track best model
                if metrics["f1_score"] > best_f1:
                    best_f1 = metrics["f1_score"]
                    best_config = config
                    log_model(
                        model=model,
                        artifact_path="model",
                        registered_model_name="random_forest_best",
                    )

                logger.info(
                    "Configuration tested",
                    config_id=idx,
                    f1_score=metrics["f1_score"],
                )

        # Log best configuration in parent run
        log_params({f"best_{k}": v for k, v in best_config.items()})
        log_metrics({"best_f1_score": best_f1})

        logger.info(
            "Hyperparameter tuning completed",
            best_config=best_config,
            best_f1=best_f1,
            parent_run_id=parent_run.info.run_id,
        )


def progressive_training():
    """Example 3: Track metrics over training iterations."""
    logger.info("Example 3: Progressive training with step tracking")

    # Generate data
    X_train, X_test, y_train, y_test = generate_sample_data()

    with start_run(run_name="progressive_training") as run:
        logger.info("Started progressive training", run_id=run.info.run_id)

        params = {
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
        }
        log_params(params)

        # Train models with increasing number of trees
        n_estimators_range = [10, 25, 50, 100, 200, 300]

        for step, n_trees in enumerate(n_estimators_range):
            logger.info("Training with n_estimators", n_estimators=n_trees, step=step)

            # Train model
            model = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=params["random_state"],
            )
            model.fit(X_train, y_train)

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics with step
            log_metrics(
                {
                    "n_estimators": n_trees,
                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["f1_score"],
                },
                step=step,
            )

            logger.info(
                "Step completed",
                step=step,
                n_estimators=n_trees,
                accuracy=metrics["accuracy"],
            )

        # Save final model
        log_model(model, "model", registered_model_name="random_forest_progressive")

        logger.info("Progressive training completed", run_id=run.info.run_id)


def main():
    """Run all examples."""
    logger.info("Starting MLflow tracking examples")

    try:
        # Example 1: Single model training
        train_single_model()

        logger.info("=" * 80)

        # Example 2: Hyperparameter tuning
        hyperparameter_tuning()

        logger.info("=" * 80)

        # Example 3: Progressive training
        progressive_training()

        logger.info("=" * 80)
        logger.info("All examples completed successfully!")
        logger.info("View results at http://localhost:5000")

    except Exception as e:
        logger.error("Example execution failed", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
