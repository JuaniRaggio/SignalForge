# SHAP Explainability Module

## Overview

The SHAP explainability module provides comprehensive model explanation capabilities for SignalForge's machine learning models using SHAP (SHapley Additive exPlanations) values.

## Features

- Multiple SHAP methods (Kernel, Tree, Linear, Deep)
- Single and batch prediction explanations
- Global feature importance analysis
- Human-readable explanations with financial context
- Visualization data generation (waterfall and summary plots)
- Fallback to permutation importance if SHAP unavailable

## Installation

The `shap` library is required and has been added to `pyproject.toml`:

```bash
# Install/update dependencies
pip install -e .
```

## Module Structure

```
src/signalforge/ml/inference/
├── __init__.py          # Public API exports
└── explainer.py         # Core explainability implementation

tests/
└── test_explainer.py    # Comprehensive test suite (29 tests)

examples/
└── explainer_example.py # Usage demonstration
```

## Core Components

### Dataclasses

#### FeatureImportance
Represents a single feature's contribution to a prediction:
- `feature`: Feature name
- `importance`: Absolute SHAP value (magnitude)
- `direction`: "positive" or "negative" impact

#### ExplanationResult
Complete explanation for a prediction:
- `prediction`: Model's predicted value
- `base_value`: Expected value (baseline)
- `feature_contributions`: List of FeatureImportance
- `top_features`: Most important feature names
- `summary_text`: Human-readable explanation

#### ExplainerConfig
Configuration for explainer behavior:
- `method`: SHAP method ("kernel", "tree", "linear", "deep")
- `n_samples`: Background samples for KernelSHAP (default: 100)
- `max_features`: Top features to show (default: 10)

### Classes

#### BaseExplainer (ABC)
Abstract interface defining:
- `explain(model, X)`: Explain single prediction
- `explain_batch(model, X)`: Explain batch predictions
- `get_feature_importance(model, X)`: Calculate global importance

#### ModelExplainer
Concrete SHAP-based implementation with:
- Automatic SHAP explainer initialization
- Support for all SHAP methods
- Lazy loading of explainers
- Fallback to permutation importance

### Functions

#### generate_explanation_text(result)
Generates human-readable explanations with financial context for common indicators:
- RSI (overbought/oversold conditions)
- MACD (momentum and divergence)
- Moving averages (trend strength)
- Volume (conviction)
- Bollinger Bands (strength/weakness)
- Volatility indicators

#### plot_waterfall(explanation)
Returns data dict for waterfall chart showing feature contributions:
- `features`: Feature names
- `values`: Signed SHAP values
- `base_value`: Expected value
- `prediction`: Final prediction

#### plot_summary(explanations)
Returns data dict for summary plot across multiple predictions:
- `features`: Unique feature names
- `importances`: Mean absolute importance per feature
- `shap_values`: 2D array of SHAP values

## Usage Examples

### Basic Usage

```python
from signalforge.ml.inference import ModelExplainer, ExplainerConfig
import polars as pl

# Configure explainer
config = ExplainerConfig(method="tree", max_features=10)
explainer = ModelExplainer(config)

# Explain a single prediction
X = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
result = explainer.explain(model, X)

print(result.summary_text)
print(f"Prediction: {result.prediction:.4f}")
print(f"Top features: {result.top_features}")
```

### Batch Processing

```python
# Explain multiple predictions
X_batch = pl.DataFrame({
    "rsi_14": [65.0, 45.0, 55.0],
    "macd": [0.5, -0.3, 0.1]
})

results = explainer.explain_batch(model, X_batch)

for i, result in enumerate(results, 1):
    print(f"Prediction {i}: {result.prediction:.4f}")
    print(f"Top feature: {result.top_features[0]}")
```

### Global Feature Importance

```python
# Calculate feature importance across dataset
importances = explainer.get_feature_importance(model, X_batch)

for imp in importances[:5]:
    print(f"{imp.feature}: {imp.importance:.4f} ({imp.direction})")
```

### Visualization Data

```python
from signalforge.ml.inference import plot_waterfall, plot_summary

# Waterfall plot data
waterfall_data = plot_waterfall(result)

# Summary plot data
summary_data = plot_summary(results)

# Use with plotting library of choice (matplotlib, plotly, etc.)
```

## SHAP Methods

### KernelSHAP (method="kernel")
- Model-agnostic method
- Works with any model type
- Slower but most flexible
- Default method

### TreeSHAP (method="tree")
- For tree-based models (XGBoost, LightGBM, RandomForest)
- Very fast and exact
- Recommended for tree models

### LinearSHAP (method="linear")
- For linear models
- Fast computation
- Works with linear regression, logistic regression

### DeepSHAP (method="deep")
- For neural networks
- Combines DeepLIFT and SHAP
- Requires PyTorch/TensorFlow models

## Error Handling

The explainer includes comprehensive error handling:

- **ImportError**: Raised if SHAP library not installed
- **ValueError**: Raised for invalid inputs (empty DataFrames, wrong dimensions)
- **RuntimeError**: Raised for computation failures

The module automatically falls back to permutation importance if SHAP fails.

## Testing

Comprehensive test suite with 29 tests covering:
- Dataclass creation and validation
- Single and batch explanations
- Feature importance calculation
- Text generation with financial context
- Visualization data generation
- Error handling and edge cases
- Integration tests

All tests use mocked SHAP to avoid dependency in test environment.

Run tests:
```bash
pytest tests/test_explainer.py -v
```

## Type Safety

The module is fully typed with strict mypy compliance:
- All functions have complete type annotations
- Literal types for direction and method enums
- Proper handling of Optional types
- No type: ignore comments needed

Verify types:
```bash
mypy src/signalforge/ml/inference/explainer.py --strict
```

## Code Quality

The module follows SignalForge's strict code quality standards:
- Passes ruff linting with all checks
- Comprehensive docstrings (Google style)
- Clear separation of concerns
- Single Responsibility Principle
- Proper error handling and logging

## Integration with SignalForge

The explainer integrates seamlessly with:
- SignalForge's structured logging (structlog)
- Polars DataFrames (not pandas)
- Existing model infrastructure (BasePredictor)
- MLflow tracking (models can be logged)

## Performance Considerations

- SHAP computation can be expensive for large datasets
- Use `n_samples` parameter to control background data size
- TreeSHAP is much faster than KernelSHAP for tree models
- Consider caching explanations for repeated queries
- Batch processing is more efficient than individual explanations

## Future Enhancements

Potential improvements:
- Caching mechanism for repeated explanations
- Interactive visualizations (plotly integration)
- Explanation comparison across models
- Time series specific explanations
- Explanations for ensemble models
- Custom financial indicator contexts
