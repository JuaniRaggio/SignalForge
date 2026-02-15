"""Example usage of Quantile Regression for prediction intervals.

This example demonstrates how to use the Quantile Regression module to:
1. Generate prediction intervals for financial time series
2. Evaluate interval coverage and calibration
3. Compare linear vs gradient boosting approaches
4. Visualize uncertainty in predictions

Run this example:
    python examples/quantile_regression_example.py
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from signalforge.ml.models.quantile_regression import (
    QuantileGradientBoostingRegressor,
    QuantileRegressor,
    QuantileRegressionConfig,
    calculate_coverage,
)


def create_sample_data(n_days: int = 200) -> pl.DataFrame:
    """Create synthetic OHLCV data with trend and volatility.

    Args:
        n_days: Number of days of data to generate.

    Returns:
        DataFrame with OHLCV data and technical indicators.
    """
    np.random.seed(42)

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Generate price series with trend, seasonality, and volatility
    base = 100.0
    trend = np.linspace(0, 30, n_days)
    seasonal = 5 * np.sin(np.linspace(0, 4 * np.pi, n_days))
    volatility = np.random.normal(0, 2, n_days)
    close_prices = base + trend + seasonal + volatility

    # Ensure positive prices
    close_prices = np.maximum(close_prices, 50.0)

    # Generate OHLC from close
    open_prices = close_prices * (1 + np.random.uniform(-0.02, 0.02, n_days))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.03, n_days))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.03, n_days))
    volumes = np.random.randint(1000000, 5000000, n_days)

    # Simple technical indicators
    sma_20 = np.convolve(close_prices, np.ones(20) / 20, mode="same")
    rsi = 50 + 20 * np.sin(np.linspace(0, 8 * np.pi, n_days))

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
            "sma_20": sma_20,
            "rsi": rsi,
        }
    )


def example_basic_quantile_regression() -> None:
    """Example 1: Basic quantile regression with 80% prediction interval."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Linear Quantile Regression")
    print("=" * 80)

    # Generate data
    df = create_sample_data(n_days=150)

    # Split into train/test
    train_df = df.head(120)
    test_df = df.tail(30)

    # Configure model for 80% prediction interval (0.1 to 0.9 quantiles)
    config = QuantileRegressionConfig(
        quantiles=[0.1, 0.5, 0.9],  # Lower, median, upper
        alpha=0.01,  # Light regularization
        n_lags=5,  # Use 5 lag features
    )

    # Create and train model
    model = QuantileRegressor(config)
    print(f"\nTraining on {train_df.height} samples...")
    model.fit(train_df, target_column="close")

    # Generate predictions
    horizon = 10
    predictions = model.predict(horizon=horizon)

    print(f"\nPredictions for next {horizon} days:")
    print(predictions)

    # Evaluate on test set
    metrics = model.evaluate(test_df)
    print("\nEvaluation Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Expected Coverage: {metrics['expected_coverage']:.1%}")
    print(f"  Empirical Coverage: {metrics['empirical_coverage']:.1%}")
    print(f"  Coverage Deviation: {metrics['coverage_deviation']:.1%}")
    print(f"  Winkler Score: {metrics['winkler_score']:.4f}")
    print(f"  Avg Interval Width: {metrics['interval_width']:.4f}")


def example_gradient_boosting() -> None:
    """Example 2: Gradient boosting for non-linear patterns."""
    print("\n" + "=" * 80)
    print("Example 2: Gradient Boosting Quantile Regression")
    print("=" * 80)

    # Generate data
    df = create_sample_data(n_days=150)

    # Split data
    train_df = df.head(120)
    test_df = df.tail(30)

    # Configure with wider intervals
    config = QuantileRegressionConfig(
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],  # Multiple quantiles
        n_lags=3,
    )

    # Create gradient boosting model
    model = QuantileGradientBoostingRegressor(
        config,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    print(f"\nTraining gradient boosting model on {train_df.height} samples...")
    model.fit(train_df)

    # Predictions
    predictions = model.predict(horizon=5)
    print("\nPredictions with multiple quantiles:")
    print(predictions)

    # Evaluate
    metrics = model.evaluate(test_df)
    print("\nEvaluation Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Empirical Coverage: {metrics['empirical_coverage']:.1%}")
    print(f"  Expected Coverage: {metrics['expected_coverage']:.1%}")


def example_coverage_calibration() -> None:
    """Example 3: Check prediction interval calibration."""
    print("\n" + "=" * 80)
    print("Example 3: Prediction Interval Calibration")
    print("=" * 80)

    # Generate data
    df = create_sample_data(n_days=200)
    train_df = df.head(150)
    test_df = df.tail(50)

    # Try different confidence levels
    confidence_levels = [0.5, 0.8, 0.9, 0.95]

    print("\nTesting different confidence levels:")
    print(f"{'Confidence':<12} {'Expected':<12} {'Empirical':<12} {'Deviation':<12}")
    print("-" * 48)

    for conf in confidence_levels:
        alpha = (1 - conf) / 2
        lower_q = alpha
        upper_q = 1 - alpha

        config = QuantileRegressionConfig(
            quantiles=[lower_q, 0.5, upper_q], alpha=0.01, n_lags=3
        )

        model = QuantileRegressor(config)
        model.fit(train_df)

        # Evaluate
        metrics = model.evaluate(test_df)
        expected = metrics["expected_coverage"]
        empirical = metrics["empirical_coverage"]
        deviation = metrics["coverage_deviation"]

        print(
            f"{conf:.0%}         {expected:.1%}        {empirical:.1%}        {deviation:.1%}"
        )


def example_custom_features() -> None:
    """Example 4: Using custom features."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Feature Selection")
    print("=" * 80)

    df = create_sample_data(n_days=150)
    train_df = df.head(120)

    # Specify custom features
    config = QuantileRegressionConfig(
        quantiles=[0.1, 0.5, 0.9],
        features=["open", "high", "low", "volume", "sma_20"],  # Custom features
        n_lags=3,
        alpha=0.05,
    )

    model = QuantileRegressor(config)
    print(f"\nTraining with custom features: {config.features}")
    model.fit(train_df)

    predictions = model.predict(horizon=7)
    print("\nPredictions:")
    print(predictions.select(["timestamp", "prediction", "lower_bound", "upper_bound"]))


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 80)
    print("QUANTILE REGRESSION EXAMPLES FOR SIGNALFORGE")
    print("=" * 80)

    example_basic_quantile_regression()
    example_gradient_boosting()
    example_coverage_calibration()
    example_custom_features()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
