# Quantile Regression Module Guide

## Overview

The Quantile Regression module provides prediction interval generation for financial time series forecasting. Unlike traditional point forecasts, quantile regression produces confidence intervals that quantify uncertainty in predictions.

## Key Features

- Multiple quantile predictions for comprehensive uncertainty quantification
- Linear and non-linear (gradient boosting) implementations
- Automatic feature selection from OHLCV and technical indicators
- Calibration metrics for interval quality assessment
- Integration with MLflow for experiment tracking
- Full compatibility with SignalForge's BasePredictor interface

## Installation

Ensure scikit-learn is installed (should be included in SignalForge dependencies):

```bash
pip install scikit-learn>=1.3.0
```

## Quick Start

```python
from signalforge.ml.models.quantile_regression import (
    QuantileRegressor,
    QuantileRegressionConfig
)

# Configure model for 80% prediction interval
config = QuantileRegressionConfig(
    quantiles=[0.1, 0.5, 0.9],  # Lower, median, upper
    alpha=0.01,                  # Regularization strength
    n_lags=5                     # Number of lag features
)

# Create and train model
model = QuantileRegressor(config)
model.fit(df, target_column="close")

# Generate predictions with intervals
predictions = model.predict(horizon=10)
```

## Core Components

### 1. QuantileRegressionConfig

Configuration dataclass for model hyperparameters.

**Parameters:**
- `quantiles`: List of quantiles to predict (e.g., `[0.1, 0.5, 0.9]`)
- `alpha`: L1 regularization strength (default: 0.1)
- `max_iter`: Maximum solver iterations (default: 1000)
- `features`: Feature columns to use (None = auto-select)
- `solver`: Optimization solver ('highs-ds', 'highs-ipm', 'highs')
- `n_lags`: Number of lag features to create (default: 5)

**Example:**
```python
config = QuantileRegressionConfig(
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    alpha=0.05,
    max_iter=2000,
    features=["open", "high", "low", "volume", "sma_20"],
    n_lags=7
)
```

### 2. QuantileRegressor

Linear quantile regression using sklearn's QuantileRegressor.

**Advantages:**
- Fast training and prediction
- Robust to outliers
- Interpretable coefficients
- Good for linear relationships

**Methods:**
- `fit(df, target_column)`: Train model on historical data
- `predict(horizon)`: Generate multi-step predictions with intervals
- `evaluate(test_df)`: Compute interval quality metrics
- `is_fitted`: Check if model is trained

**Example:**
```python
model = QuantileRegressor(config)
model.fit(train_df)

# Predict next 10 days
preds = model.predict(horizon=10)
print(preds.select(["timestamp", "prediction", "lower_bound", "upper_bound"]))

# Evaluate on test set
metrics = model.evaluate(test_df)
print(f"Coverage: {metrics['empirical_coverage']:.1%}")
```

### 3. QuantileGradientBoostingRegressor

Gradient boosting implementation for non-linear patterns.

**Advantages:**
- Captures non-linear relationships
- Better accuracy on complex data
- Feature importance analysis

**Additional Parameters:**
- `n_estimators`: Number of boosting stages (default: 100)
- `learning_rate`: Step size shrinkage (default: 0.1)
- `max_depth`: Maximum tree depth (default: 3)
- `subsample`: Fraction of samples per tree (default: 1.0)
- `random_state`: Random seed for reproducibility

**Example:**
```python
model = QuantileGradientBoostingRegressor(
    config,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(train_df)
predictions = model.predict(horizon=20)
```

### 4. QuantilePrediction

Structured prediction output with intervals.

**Attributes:**
- `point_forecast`: Median or mean prediction
- `lower_bound`: Lower prediction bound
- `upper_bound`: Upper prediction bound
- `all_quantiles`: Dict of all quantile predictions
- `coverage`: Expected coverage probability
- `timestamp`: Optional prediction timestamp

### 5. Helper Functions

#### create_quantile_regressor()

Factory function for model creation.

```python
from signalforge.ml.models.quantile_regression import create_quantile_regressor

model = create_quantile_regressor(config, method="linear")  # or "gbm"
```

#### calculate_coverage()

Compute empirical coverage of prediction intervals.

```python
from signalforge.ml.models.quantile_regression import calculate_coverage

coverage = calculate_coverage(predictions, actuals)
print(f"Coverage: {coverage:.1%}")
```

#### winkler_score()

Calculate interval quality score (lower is better).

```python
from signalforge.ml.models.quantile_regression import winkler_score
import numpy as np

score = winkler_score(
    lower=np.array([95.0, 98.0]),
    upper=np.array([105.0, 108.0]),
    actual=np.array([100.0, 110.0]),
    alpha=0.2  # 1 - coverage
)
```

## Evaluation Metrics

The `evaluate()` method returns comprehensive metrics:

**Point Forecast Metrics:**
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `mape`: Mean Absolute Percentage Error

**Interval Quality Metrics:**
- `empirical_coverage`: Actual fraction of observations in intervals
- `expected_coverage`: Theoretical coverage (upper_quantile - lower_quantile)
- `coverage_deviation`: |empirical - expected| (should be small)
- `winkler_score`: Penalizes wide intervals and violations
- `interval_width`: Average prediction interval width

**Interpretation:**
- Well-calibrated intervals have low coverage deviation
- Narrower intervals (when accurate) are better than wider ones
- Winkler score balances coverage and width

## Best Practices

### 1. Choosing Quantiles

```python
# Standard 80% interval
quantiles=[0.1, 0.5, 0.9]

# Conservative 90% interval
quantiles=[0.05, 0.5, 0.95]

# Multiple intervals for detailed uncertainty
quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
```

### 2. Feature Selection

Auto-selection works well but custom features can improve performance:

```python
# Auto-select from OHLCV + technical indicators
config = QuantileRegressionConfig(quantiles=[0.1, 0.5, 0.9])

# Custom features for specific use case
config = QuantileRegressionConfig(
    quantiles=[0.1, 0.5, 0.9],
    features=["open", "high", "low", "volume", "rsi", "macd", "bb_upper"]
)
```

### 3. Lag Features

More lags capture longer history but require more data:

```python
# Short-term dependencies
n_lags=3

# Medium-term (default)
n_lags=5

# Long-term patterns
n_lags=10
```

### 4. Model Selection

Choose based on your data characteristics:

```python
# Linear model: fast, interpretable, good for trending data
model = QuantileRegressor(config)

# Gradient boosting: better for complex patterns, needs more data
model = QuantileGradientBoostingRegressor(
    config,
    n_estimators=100,
    max_depth=3
)
```

### 5. Regularization

Balance between overfitting and underfitting:

```python
# Light regularization (more complex model)
alpha=0.01

# Moderate (default)
alpha=0.1

# Strong (simpler model)
alpha=1.0
```

## Common Use Cases

### 1. Risk Management

Generate prediction intervals to assess downside risk:

```python
config = QuantileRegressionConfig(
    quantiles=[0.05, 0.5, 0.95],  # 90% confidence
    alpha=0.01
)
model = QuantileRegressor(config)
model.fit(historical_data)

predictions = model.predict(horizon=5)
worst_case = predictions["quantile_0.05"].min()
print(f"5th percentile outcome: ${worst_case:.2f}")
```

### 2. Trading Strategy Evaluation

Assess if predictions are well-calibrated:

```python
# Train model
model.fit(train_df)

# Evaluate on out-of-sample data
metrics = model.evaluate(test_df)

# Check calibration
if metrics["coverage_deviation"] < 0.1:
    print("Model is well-calibrated!")
else:
    print("Model needs recalibration")
```

### 3. Portfolio Optimization

Use prediction intervals for scenario analysis:

```python
predictions = model.predict(horizon=20)

# Extract quantile scenarios
scenarios = {
    "pessimistic": predictions["quantile_0.1"],
    "base": predictions["quantile_0.5"],
    "optimistic": predictions["quantile_0.9"]
}

# Use in portfolio optimization
```

## Troubleshooting

### Issue: Wide Prediction Intervals

**Causes:**
- High data volatility
- Insufficient features
- Too much regularization

**Solutions:**
```python
# Add more informative features
config.features = ["open", "high", "low", "volume", "sma_20", "rsi", "macd"]

# Reduce regularization
config.alpha = 0.01

# Try gradient boosting for complex patterns
model = QuantileGradientBoostingRegressor(config, n_estimators=200)
```

### Issue: Poor Coverage Calibration

**Causes:**
- Insufficient training data
- Data distribution shift
- Wrong quantile specification

**Solutions:**
```python
# Ensure sufficient data (100+ samples recommended)
assert len(train_df) >= 100

# Check for outliers or regime changes
# Consider training on more recent data

# Verify quantile configuration
config = QuantileRegressionConfig(
    quantiles=[0.1, 0.5, 0.9]  # Should give 80% coverage
)
```

### Issue: Quantile Crossing

When lower quantile exceeds upper quantile (rare but possible):

```python
# Use sorted quantiles
config.quantiles = sorted([0.1, 0.5, 0.9])

# For gradient boosting, increase n_estimators
model = QuantileGradientBoostingRegressor(
    config,
    n_estimators=200,  # More trees help
    learning_rate=0.05  # Slower learning
)
```

## Performance Considerations

### Training Time

- **Linear QR**: Fast (seconds for 1000s of samples)
- **Gradient Boosting**: Slower (minutes for large datasets)

Scale with:
- Number of quantiles
- Number of features
- Number of samples
- For GBM: n_estimators, max_depth

### Memory Usage

- **Linear QR**: Low (stores coefficients only)
- **Gradient Boosting**: Higher (stores trees)

Optimize by:
- Reducing number of quantiles
- Feature selection
- For GBM: Lower max_depth

### Prediction Speed

Both models are fast for inference (milliseconds for typical horizons).

## Integration with MLflow

The module automatically logs to MLflow when an active run exists:

```python
from signalforge.ml.training.mlflow_config import start_run

with start_run(run_name="quantile_regression_experiment"):
    model = QuantileRegressor(config)
    model.fit(train_df)  # Params logged automatically

    metrics = model.evaluate(test_df)  # Metrics logged automatically
    predictions = model.predict(horizon=10)
```

Without an active run, logging is skipped gracefully (no errors).

## References

- Koenker, R. (2005). "Quantile Regression". Cambridge University Press.
- sklearn QuantileRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
- Gradient Boosting for Quantile Regression: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html

## Example Scripts

See `/examples/quantile_regression_example.py` for complete working examples including:
- Basic linear quantile regression
- Gradient boosting implementation
- Coverage calibration testing
- Custom feature selection
