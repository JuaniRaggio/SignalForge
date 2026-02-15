"""Example usage of the SHAP explainability module.

This script demonstrates how to use the ModelExplainer to explain
model predictions with SHAP values, including:
- Single prediction explanation
- Batch prediction explanation
- Global feature importance
- Visualization data generation
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from signalforge.ml.inference import ExplainerConfig, ModelExplainer


def main() -> None:
    """Demonstrate explainer functionality."""
    print("SHAP Explainability Example\n")
    print("=" * 70)

    # Create sample data
    print("\n1. Creating sample dataset...")
    np.random.seed(42)

    n_samples = 100
    df = pl.DataFrame(
        {
            "rsi_14": np.random.uniform(30, 70, n_samples),
            "macd": np.random.uniform(-1, 1, n_samples),
            "sma_20": np.random.uniform(90, 110, n_samples),
            "volume": np.random.uniform(500000, 1500000, n_samples),
            "target": np.random.uniform(0, 1, n_samples),
        }
    )

    print(f"Created dataset with {n_samples} samples and {len(df.columns)-1} features")

    # Train a simple model
    print("\n2. Training Random Forest model...")
    X = df.select(["rsi_14", "macd", "sma_20", "volume"]).to_numpy()
    y = df["target"].to_numpy()

    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    print("Model trained successfully")

    # Create explainer
    print("\n3. Initializing SHAP explainer...")
    config = ExplainerConfig(method="tree", n_samples=50, max_features=5)
    explainer = ModelExplainer(config)
    print(f"Explainer configured with method={config.method}")

    # Single prediction explanation
    print("\n4. Explaining single prediction...")
    print("-" * 70)

    single_sample = df.select(["rsi_14", "macd", "sma_20", "volume"]).head(1)
    print(f"\nInput features:")
    print(single_sample)

    result = explainer.explain(model, single_sample)

    print(f"\nPrediction: {result.prediction:.4f}")
    print(f"Base value: {result.base_value:.4f}")
    print(f"\nTop {len(result.top_features)} contributing features:")
    for i, feature in enumerate(result.top_features, 1):
        contrib = next(fc for fc in result.feature_contributions if fc.feature == feature)
        print(
            f"  {i}. {contrib.feature}: {contrib.importance:.4f} ({contrib.direction})"
        )

    print(f"\n{result.summary_text}")

    # Batch prediction explanation
    print("\n5. Explaining batch predictions...")
    print("-" * 70)

    batch_sample = df.select(["rsi_14", "macd", "sma_20", "volume"]).head(5)
    print(f"\nExplaining {batch_sample.height} predictions...")

    batch_results = explainer.explain_batch(model, batch_sample)

    for i, result in enumerate(batch_results, 1):
        print(f"\nPrediction {i}:")
        print(f"  Value: {result.prediction:.4f}")
        print(f"  Top feature: {result.top_features[0]}")

    # Global feature importance
    print("\n6. Computing global feature importance...")
    print("-" * 70)

    sample_data = df.select(["rsi_14", "macd", "sma_20", "volume"]).head(30)
    importances = explainer.get_feature_importance(model, sample_data)

    print(f"\nGlobal feature importance (sorted by impact):")
    for i, imp in enumerate(importances, 1):
        print(f"  {i}. {imp.feature}: {imp.importance:.4f} ({imp.direction})")

    # Generate visualization data
    print("\n7. Generating visualization data...")
    print("-" * 70)

    from signalforge.ml.inference import plot_summary, plot_waterfall

    # Waterfall plot data
    waterfall_data = plot_waterfall(batch_results[0])
    print(f"\nWaterfall plot data for first prediction:")
    print(f"  Base value: {waterfall_data['base_value']:.4f}")
    print(f"  Prediction: {waterfall_data['prediction']:.4f}")
    print(f"  Features: {waterfall_data['features']}")
    print(f"  Values: {[f'{v:.4f}' for v in waterfall_data['values']]}")

    # Summary plot data
    summary_data = plot_summary(batch_results)
    print(f"\nSummary plot data for batch:")
    print(f"  Features: {summary_data['features']}")
    print(f"  Mean importance: {[f'{v:.4f}' for v in summary_data['importances']]}")
    print(f"  SHAP values shape: {len(summary_data['shap_values'])} samples x {len(summary_data['features'])} features")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("\nNote: This example uses tree-based SHAP for speed.")
    print("For other models, use method='kernel' in ExplainerConfig.")


if __name__ == "__main__":
    main()
