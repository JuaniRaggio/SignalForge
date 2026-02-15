"""Example usage of the execution quality module.

This script demonstrates how to assess liquidity and estimate slippage
for trading signals using real market data structure.
"""

from datetime import datetime, timedelta

import polars as pl

from signalforge.execution import assess_liquidity, estimate_slippage


def create_sample_data(n_days: int = 30) -> pl.DataFrame:
    """Create sample OHLCV data for demonstration."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": [150.0 + i * 0.5 for i in range(n_days)],
            "high": [155.0 + i * 0.5 for i in range(n_days)],
            "low": [148.0 + i * 0.5 for i in range(n_days)],
            "close": [152.0 + i * 0.5 for i in range(n_days)],
            "volume": [5_000_000 + i * 10_000 for i in range(n_days)],
        }
    )


def main() -> None:
    """Run execution quality analysis examples."""
    print("=" * 80)
    print("SignalForge - Execution Quality Analysis")
    print("=" * 80)
    print()

    # Create sample data
    df = create_sample_data()
    symbol = "AAPL"

    # Example 1: Assess Liquidity
    print("Example 1: Liquidity Assessment")
    print("-" * 80)

    liquidity = assess_liquidity(df, symbol)

    print(f"Symbol: {liquidity.symbol}")
    print(f"Average Daily Volume: {liquidity.avg_daily_volume:,.0f} shares")
    print(f"Volume Volatility: {liquidity.volume_volatility:,.0f} shares")
    print(f"Liquidity Score: {liquidity.liquidity_score:.2f}/100")
    print(f"Is Liquid: {'Yes' if liquidity.is_liquid else 'No'}")

    if liquidity.liquidity_score > 70:
        print("Assessment: HIGHLY LIQUID - Suitable for most strategies")
    elif liquidity.liquidity_score > 40:
        print("Assessment: MEDIUM LIQUIDITY - Use with caution")
    else:
        print("Assessment: LOW LIQUIDITY - High execution risk")

    print()

    # Example 2: Estimate Slippage for Small Order
    print("Example 2: Slippage Estimation - Small Order")
    print("-" * 80)

    current_price = float(df["close"][-1])
    small_order = 25_000  # $25K order

    slippage_small = estimate_slippage(
        order_size_usd=small_order,
        avg_daily_volume=liquidity.avg_daily_volume,
        current_price=current_price,
        volatility=0.02,  # 2% daily volatility
        symbol=symbol,
    )

    print(f"Order Size: ${slippage_small.order_size:,.2f}")
    print(f"ADV Ratio: {slippage_small.adv_ratio * 100:.4f}%")
    print(f"Estimated Slippage: {slippage_small.estimated_slippage_pct:.4f}%")
    print(f"Estimated Cost: ${slippage_small.estimated_slippage_usd:.2f}")
    print(f"Execution Risk: {slippage_small.execution_risk.upper()}")

    print()

    # Example 3: Estimate Slippage for Large Order
    print("Example 3: Slippage Estimation - Large Order")
    print("-" * 80)

    large_order = 500_000  # $500K order

    slippage_large = estimate_slippage(
        order_size_usd=large_order,
        avg_daily_volume=liquidity.avg_daily_volume,
        current_price=current_price,
        volatility=0.02,
        symbol=symbol,
    )

    print(f"Order Size: ${slippage_large.order_size:,.2f}")
    print(f"ADV Ratio: {slippage_large.adv_ratio * 100:.4f}%")
    print(f"Estimated Slippage: {slippage_large.estimated_slippage_pct:.4f}%")
    print(f"Estimated Cost: ${slippage_large.estimated_slippage_usd:.2f}")
    print(f"Execution Risk: {slippage_large.execution_risk.upper()}")

    print()

    # Example 4: Compare High vs Low Volatility
    print("Example 4: Volatility Impact on Slippage")
    print("-" * 80)

    order_size = 100_000

    slippage_low_vol = estimate_slippage(
        order_size_usd=order_size,
        avg_daily_volume=liquidity.avg_daily_volume,
        current_price=current_price,
        volatility=0.01,  # Low volatility (1%)
        symbol=symbol,
    )

    slippage_high_vol = estimate_slippage(
        order_size_usd=order_size,
        avg_daily_volume=liquidity.avg_daily_volume,
        current_price=current_price,
        volatility=0.05,  # High volatility (5%)
        symbol=symbol,
    )

    print(f"Order Size: ${order_size:,.2f}")
    print()
    print("Low Volatility (1%):")
    print(f"  Slippage: {slippage_low_vol.estimated_slippage_pct:.4f}%")
    print(f"  Cost: ${slippage_low_vol.estimated_slippage_usd:.2f}")
    print()
    print("High Volatility (5%):")
    print(f"  Slippage: {slippage_high_vol.estimated_slippage_pct:.4f}%")
    print(f"  Cost: ${slippage_high_vol.estimated_slippage_usd:.2f}")
    print()
    print(
        f"Volatility Impact: {slippage_high_vol.estimated_slippage_usd / slippage_low_vol.estimated_slippage_usd:.2f}x higher cost"
    )

    print()

    # Example 5: Trading Signal Validation
    print("Example 5: Trading Signal Validation Workflow")
    print("-" * 80)

    signal_order_size = 150_000
    max_slippage_pct = 0.5  # Maximum acceptable slippage: 0.5%
    min_liquidity_score = 50  # Minimum liquidity score

    print(f"Signal: BUY {symbol} - ${signal_order_size:,.2f}")
    print()

    # Check liquidity
    print("Step 1: Liquidity Check")
    if liquidity.liquidity_score >= min_liquidity_score:
        print(f"  PASS - Liquidity Score: {liquidity.liquidity_score:.2f} >= {min_liquidity_score}")
    else:
        print(f"  FAIL - Liquidity Score: {liquidity.liquidity_score:.2f} < {min_liquidity_score}")
        print("  Signal REJECTED: Insufficient liquidity")
        return

    # Check slippage
    print()
    print("Step 2: Slippage Check")
    slippage = estimate_slippage(
        order_size_usd=signal_order_size,
        avg_daily_volume=liquidity.avg_daily_volume,
        current_price=current_price,
        volatility=0.02,
        symbol=symbol,
    )

    if slippage.estimated_slippage_pct <= max_slippage_pct:
        print(
            f"  PASS - Estimated Slippage: {slippage.estimated_slippage_pct:.4f}% <= {max_slippage_pct}%"
        )
    else:
        print(
            f"  FAIL - Estimated Slippage: {slippage.estimated_slippage_pct:.4f}% > {max_slippage_pct}%"
        )
        print("  Signal REJECTED: Excessive slippage")
        return

    # Check execution risk
    print()
    print("Step 3: Execution Risk Check")
    if slippage.execution_risk in ("low", "medium"):
        print(f"  PASS - Execution Risk: {slippage.execution_risk.upper()}")
    else:
        print(f"  WARNING - Execution Risk: {slippage.execution_risk.upper()}")
        print("  Consider splitting order or using algorithmic execution")

    print()
    print("=" * 80)
    print("SIGNAL APPROVED FOR EXECUTION")
    print(f"Expected Total Cost: ${signal_order_size + slippage.estimated_slippage_usd:,.2f}")
    print(f"  Principal: ${signal_order_size:,.2f}")
    print(f"  Slippage: ${slippage.estimated_slippage_usd:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
