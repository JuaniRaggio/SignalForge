"""Example usage of BenchmarkComparator for strategy evaluation.

This example demonstrates how to use the BenchmarkComparator to evaluate
a trading strategy against market benchmarks.
"""

from decimal import Decimal

import polars as pl

from signalforge.benchmark.comparator import BenchmarkComparator


def main() -> None:
    """Demonstrate benchmark comparison functionality."""
    # Initialize comparator with 2% annual risk-free rate
    comparator = BenchmarkComparator(risk_free_rate=0.02)

    # Add S&P 500 (SPY) benchmark
    spy_returns = pl.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.02, 0.01])
    spy_equity = [
        Decimal("10000"),
        Decimal("10100"),
        Decimal("10302"),
        Decimal("10199"),
        Decimal("10352"),
        Decimal("10300"),
        Decimal("10506"),
        Decimal("10611"),
    ]
    comparator.add_benchmark("SPY", spy_returns, spy_equity)

    # Add Nasdaq 100 (QQQ) benchmark
    qqq_returns = pl.Series([0.015, 0.025, -0.015, 0.02, -0.01, 0.025, 0.015])
    qqq_equity = [
        Decimal("10000"),
        Decimal("10150"),
        Decimal("10404"),
        Decimal("10248"),
        Decimal("10453"),
        Decimal("10349"),
        Decimal("10607"),
        Decimal("10766"),
    ]
    comparator.add_benchmark("QQQ", qqq_returns, qqq_equity)

    # Strategy returns (slightly outperforming)
    strategy_returns = pl.Series([0.012, 0.022, -0.008, 0.018, -0.003, 0.023, 0.013])

    print("=== Benchmark Comparison Results ===\n")

    # Compare against SPY
    print("Comparison vs SPY:")
    print("-" * 50)
    spy_result = comparator.compare(strategy_returns, "SPY", "TestStrategy")
    print(f"Alpha: {spy_result.alpha:.2f}%")
    print(f"Beta: {spy_result.beta:.2f}")
    print(f"Correlation: {spy_result.correlation:.2f}")
    print(f"Tracking Error: {spy_result.tracking_error:.2f}%")
    print(f"Information Ratio: {spy_result.information_ratio:.2f}")
    print(f"Up Capture: {spy_result.up_capture:.1f}%")
    print(f"Down Capture: {spy_result.down_capture:.1f}%")
    print()

    # Compare against all benchmarks
    print("Comparison vs All Benchmarks:")
    print("-" * 50)
    all_results = comparator.compare_all(strategy_returns, "TestStrategy")
    for result in all_results:
        print(f"\n{result.benchmark_name}:")
        print(f"  Alpha: {result.alpha:.2f}%")
        print(f"  Beta: {result.beta:.2f}")
        print(f"  Information Ratio: {result.information_ratio:.2f}")

    # Find best performing comparison
    best_result = max(all_results, key=lambda r: r.information_ratio)
    print(f"\nBest Information Ratio vs {best_result.benchmark_name}: "
          f"{best_result.information_ratio:.2f}")

    # Individual metric calculations
    print("\n=== Individual Metric Calculations ===\n")

    beta = comparator.calculate_beta(strategy_returns, spy_returns)
    print(f"Beta vs SPY: {beta:.2f}")

    alpha = comparator.calculate_alpha(strategy_returns, spy_returns)
    print(f"Alpha vs SPY: {alpha:.2f}%")

    correlation = comparator.calculate_correlation(strategy_returns, spy_returns)
    print(f"Correlation vs SPY: {correlation:.2f}")

    tracking_error = comparator.calculate_tracking_error(strategy_returns, spy_returns)
    print(f"Tracking Error vs SPY: {tracking_error:.2f}%")

    up_capture, down_capture = comparator.calculate_capture_ratios(
        strategy_returns, spy_returns
    )
    print(f"Up Capture vs SPY: {up_capture:.1f}%")
    print(f"Down Capture vs SPY: {down_capture:.1f}%")


if __name__ == "__main__":
    main()
