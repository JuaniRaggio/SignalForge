# ML Inference and Explainability

This module provides model inference capabilities and SHAP-based explainability for SignalForge.

## Quick Start

```python
from signalforge.ml.inference import ModelExplainer, ExplainerConfig
import polars as pl

# Configure and create explainer
config = ExplainerConfig(method="tree", max_features=10)
explainer = ModelExplainer(config)

# Explain a prediction
X = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5], "volume": [1000000.0]})
result = explainer.explain(model, X)

# View explanation
print(result.summary_text)
print(f"Prediction: {result.prediction:.4f}")
print(f"Top features: {result.top_features}")

# Feature contributions
for contrib in result.feature_contributions[:3]:
    print(f"{contrib.feature}: {contrib.importance:.4f} ({contrib.direction})")
```

## API Reference

### Configuration

```python
ExplainerConfig(
    method="kernel",      # SHAP method: "kernel", "tree", "linear", "deep"
    n_samples=100,        # Background samples for KernelSHAP
    max_features=10       # Top features to show in explanations
)
```

### Methods

#### explain(model, X)
Explain single prediction (X must have exactly 1 row).

Returns `ExplanationResult` with:
- `prediction`: Model output
- `base_value`: Expected value
- `feature_contributions`: List of FeatureImportance
- `top_features`: Most important features
- `summary_text`: Human-readable explanation

#### explain_batch(model, X)
Explain multiple predictions (X can have multiple rows).

Returns list of `ExplanationResult`, one per row.

#### get_feature_importance(model, X)
Calculate global feature importance across samples.

Returns list of `FeatureImportance` sorted by importance.

### Visualization

```python
from signalforge.ml.inference import plot_waterfall, plot_summary

# Waterfall plot data (for single prediction)
waterfall_data = plot_waterfall(result)

# Summary plot data (for batch predictions)
summary_data = plot_summary(results)
```

## SHAP Methods

| Method | Model Types | Speed | Accuracy |
|--------|-------------|-------|----------|
| kernel | Any model | Slow | Approximate |
| tree | Tree-based (XGBoost, RF, LightGBM) | Very Fast | Exact |
| linear | Linear models | Fast | Exact |
| deep | Neural networks | Medium | Approximate |

## Examples

See `/examples/explainer_example.py` for complete usage demonstration.

## Testing

```bash
pytest tests/test_explainer.py -v
```

## Documentation

See `/docs/explainer_module.md` for comprehensive documentation.
