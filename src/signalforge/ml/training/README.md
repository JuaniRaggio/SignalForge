# MLflow Training Module

This module provides MLflow integration for experiment tracking, model logging, and reproducible machine learning workflows in SignalForge.

## Features

- Experiment setup and management
- Parameter logging
- Metric tracking with step support
- Model artifact logging with automatic framework detection
- Context manager for MLflow runs with proper cleanup

## Configuration

MLflow settings are configured through environment variables or in `.env` file:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=signalforge
MLFLOW_ARTIFACT_LOCATION=./mlartifacts
```

## Usage

### Basic Example

```python
from signalforge.ml.training import (
    start_run,
    log_params,
    log_metrics,
    log_model,
)

# Train a model with experiment tracking
with start_run(run_name="my_experiment") as run:
    # Log hyperparameters
    log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam",
    })

    # Training code here
    model = train_model()

    # Log metrics
    log_metrics({
        "accuracy": 0.95,
        "loss": 0.05,
        "f1_score": 0.92,
    })

    # Log the trained model
    log_model(
        model=model,
        artifact_path="model",
        registered_model_name="my_best_model"
    )
```

### Logging Metrics Over Time

```python
with start_run(run_name="training_with_steps"):
    log_params({"epochs": 100})

    for epoch in range(100):
        # Training logic
        train_loss, val_loss = train_epoch(model, data)

        # Log metrics for this epoch
        log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, step=epoch)
```

### Nested Runs

```python
# Parent run for overall experiment
with start_run(run_name="hyperparameter_tuning") as parent_run:
    log_params({"experiment_type": "grid_search"})

    for lr in [0.001, 0.01, 0.1]:
        # Nested run for each configuration
        with start_run(run_name=f"lr_{lr}", nested=True):
            log_params({"learning_rate": lr})

            # Train and evaluate
            model = train_model(learning_rate=lr)
            accuracy = evaluate_model(model)

            log_metrics({"accuracy": accuracy})
            log_model(model, "model")
```

## Supported Model Frameworks

The `log_model` function automatically detects the ML framework and uses the appropriate MLflow flavor:

- **scikit-learn**: RandomForest, SVM, LogisticRegression, etc.
- **PyTorch**: Neural networks and custom models
- **TensorFlow/Keras**: Sequential, Functional, and Subclassed models
- **XGBoost**: XGBClassifier, XGBRegressor
- **LightGBM**: LGBMClassifier, LGBMRegressor
- **Generic**: Any custom model (uses MLflow pyfunc)

## Running MLflow UI

Start the MLflow tracking server using Docker Compose:

```bash
docker-compose up -d mlflow
```

Access the UI at: http://localhost:5000

## Best Practices

### 1. Organize Experiments

Use descriptive experiment names and run names:

```python
with start_run(run_name="lstm_sentiment_analysis_v1"):
    # Your code
    pass
```

### 2. Log Comprehensive Parameters

Log all hyperparameters that affect model behavior:

```python
log_params({
    "model_type": "LSTM",
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam",
    "scheduler": "cosine",
})
```

### 3. Track Metrics at Multiple Levels

Log both training and validation metrics:

```python
log_metrics({
    "train_loss": train_loss,
    "train_accuracy": train_acc,
    "val_loss": val_loss,
    "val_accuracy": val_acc,
}, step=epoch)
```

### 4. Version Your Models

Use the model registry for production models:

```python
log_model(
    model=final_model,
    artifact_path="model",
    registered_model_name="signal_classifier_production"
)
```

### 5. Handle Errors Gracefully

The context manager ensures cleanup even on failure:

```python
try:
    with start_run(run_name="experiment"):
        # Training code that might fail
        model = risky_training_operation()
        log_model(model, "model")
except Exception as e:
    logger.error("Training failed", error=str(e))
    # MLflow run is properly closed
```

## Example: Complete Training Pipeline

```python
from signalforge.ml.training import (
    setup_experiment,
    start_run,
    log_params,
    log_metrics,
    log_model,
)
from signalforge.core.logging import get_logger

logger = get_logger(__name__)

def train_signal_classifier(config: dict):
    """Train a signal classification model with MLflow tracking."""

    # Setup experiment (only needed once)
    experiment_id = setup_experiment("signal_classification")

    with start_run(run_name=f"classifier_{config['model_type']}") as run:
        logger.info("Starting training run", run_id=run.info.run_id)

        # Log configuration
        log_params(config)

        # Prepare data
        train_data, val_data = prepare_datasets(config)

        # Initialize model
        model = create_model(config)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(config['epochs']):
            train_metrics = train_epoch(model, train_data)
            val_metrics = validate_epoch(model, val_data)

            # Log metrics
            log_metrics({
                "train_loss": train_metrics['loss'],
                "train_accuracy": train_metrics['accuracy'],
                "val_loss": val_metrics['loss'],
                "val_accuracy": val_metrics['accuracy'],
            }, step=epoch)

            # Track best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']

        # Log final metrics
        log_metrics({
            "best_val_loss": best_val_loss,
            "final_train_accuracy": train_metrics['accuracy'],
            "final_val_accuracy": val_metrics['accuracy'],
        })

        # Log trained model
        log_model(
            model=model,
            artifact_path="model",
            registered_model_name="signal_classifier"
        )

        logger.info("Training completed", run_id=run.info.run_id)

        return model, run.info.run_id
```

## Troubleshooting

### Connection Issues

If you can't connect to MLflow server:

```bash
# Check if MLflow server is running
docker-compose ps mlflow

# Check MLflow logs
docker-compose logs mlflow

# Restart MLflow server
docker-compose restart mlflow
```

### Database Issues

If you see database errors:

```bash
# Reset MLflow database (WARNING: destroys all tracking data)
docker-compose down -v
docker-compose up -d mlflow
```

## API Reference

### `setup_experiment(experiment_name: str) -> str`

Setup or get existing MLflow experiment.

**Returns:** Experiment ID

### `start_run(run_name: str | None = None, nested: bool = False)`

Context manager for MLflow runs.

**Parameters:**
- `run_name`: Optional name for the run
- `nested`: If True, creates a nested run

### `log_params(params: dict[str, Any])`

Log parameters to current run.

**Parameters:**
- `params`: Dictionary of parameter names and values

### `log_metrics(metrics: dict[str, float], step: int | None = None)`

Log metrics to current run.

**Parameters:**
- `metrics`: Dictionary of metric names and values
- `step`: Optional step number for time series tracking

### `log_model(model: Any, artifact_path: str, registered_model_name: str | None = None)`

Log model artifact with automatic framework detection.

**Parameters:**
- `model`: Trained model object
- `artifact_path`: Path within run's artifact directory
- `registered_model_name`: Optional name for model registry
