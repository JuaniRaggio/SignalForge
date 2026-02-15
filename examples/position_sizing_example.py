"""Example usage of the Position Sizing module with Kelly Criterion.

This example demonstrates how to calculate optimal position sizes using the Kelly
criterion with fractional sizing and portfolio constraints.
"""

from decimal import Decimal

from signalforge.risk import PositionSizeConfig, PositionSizer


def example_basic_position_sizing() -> None:
    """Basic example of position sizing with default settings."""
    print("=== Basic Position Sizing Example ===\n")

    # Create a position sizer with default config
    # Default: 10% max position, 25% max portfolio, half-Kelly
    sizer = PositionSizer(PositionSizeConfig())

    # Calculate position size for a profitable strategy
    result = sizer.calculate_position_size(
        portfolio_value=Decimal("100000"),  # $100,000 portfolio
        price=Decimal("50.00"),  # Stock price of $50
        win_rate=0.60,  # 60% win rate
        avg_win=Decimal("150.00"),  # Average win of $150
        avg_loss=Decimal("100.00"),  # Average loss of $100
    )

    print(f"Portfolio Value: ${result.dollar_amount:,.2f}")
    print(f"Shares to Buy: {result.shares}")
    print(f"Position Size: ${result.dollar_amount:,.2f} ({result.portfolio_pct:.1%} of portfolio)")
    print(f"Full Kelly: {result.kelly_full:.2%}")
    print(f"Adjusted Kelly (Half): {result.kelly_adjusted:.2%}")
    print(f"Risk Amount: ${result.risk_amount:,.2f}\n")


def example_aggressive_sizing() -> None:
    """Example with more aggressive settings (full Kelly, higher limits)."""
    print("=== Aggressive Position Sizing Example ===\n")

    config = PositionSizeConfig(
        max_position_pct=0.20,  # Allow up to 20% per position
        max_portfolio_pct=0.50,  # Allow up to 50% total allocation
        kelly_fraction=1.0,  # Use full Kelly (more aggressive)
        min_position_size=Decimal("1000"),  # Minimum $1,000 position
    )

    sizer = PositionSizer(config)

    result = sizer.calculate_position_size(
        portfolio_value=Decimal("100000"),
        price=Decimal("75.00"),
        win_rate=0.55,
        avg_win=Decimal("200.00"),
        avg_loss=Decimal("150.00"),
    )

    print(f"Shares to Buy: {result.shares}")
    print(f"Position Size: ${result.dollar_amount:,.2f} ({result.portfolio_pct:.1%} of portfolio)")
    print(f"Full Kelly: {result.kelly_full:.2%}")
    print(f"Risk Amount: ${result.risk_amount:,.2f}\n")


def example_conservative_sizing() -> None:
    """Example with conservative settings (quarter-Kelly, low limits)."""
    print("=== Conservative Position Sizing Example ===\n")

    config = PositionSizeConfig(
        max_position_pct=0.05,  # Max 5% per position
        max_portfolio_pct=0.15,  # Max 15% total allocation
        kelly_fraction=0.25,  # Quarter-Kelly (very conservative)
        min_position_size=Decimal("500"),  # Minimum $500 position
    )

    sizer = PositionSizer(config)

    result = sizer.calculate_position_size(
        portfolio_value=Decimal("100000"),
        price=Decimal("100.00"),
        win_rate=0.65,
        avg_win=Decimal("300.00"),
        avg_loss=Decimal("200.00"),
    )

    print(f"Shares to Buy: {result.shares}")
    print(f"Position Size: ${result.dollar_amount:,.2f} ({result.portfolio_pct:.1%} of portfolio)")
    print(f"Full Kelly: {result.kelly_full:.2%}")
    print(f"Adjusted Kelly (Quarter): {result.kelly_adjusted:.2%}")
    print(f"Risk Amount: ${result.risk_amount:,.2f}\n")


def example_losing_strategy() -> None:
    """Example with negative expectancy (should result in no position)."""
    print("=== Losing Strategy Example (Negative Expectancy) ===\n")

    sizer = PositionSizer(PositionSizeConfig())

    # Low win rate with poor win/loss ratio
    result = sizer.calculate_position_size(
        portfolio_value=Decimal("100000"),
        price=Decimal("50.00"),
        win_rate=0.40,  # Only 40% win rate
        avg_win=Decimal("100.00"),
        avg_loss=Decimal("150.00"),  # Losses bigger than wins
    )

    print(f"Shares to Buy: {result.shares}")
    print(f"Position Size: ${result.dollar_amount:,.2f}")
    print(f"Full Kelly: {result.kelly_full:.2%} (negative - don't trade!)")
    print(
        f"Status: {'NO POSITION - Negative expectancy' if result.shares == 0 else 'Position taken'}\n"
    )


def example_multiple_scenarios() -> None:
    """Compare position sizing across multiple scenarios."""
    print("=== Multiple Scenario Comparison ===\n")

    config = PositionSizeConfig(
        kelly_fraction=0.5, max_position_pct=0.15, max_portfolio_pct=0.40
    )
    sizer = PositionSizer(config)

    scenarios = [
        {
            "name": "High Win Rate, Low Ratio",
            "win_rate": 0.70,
            "avg_win": Decimal("100"),
            "avg_loss": Decimal("100"),
        },
        {
            "name": "Medium Win Rate, High Ratio",
            "win_rate": 0.50,
            "avg_win": Decimal("200"),
            "avg_loss": Decimal("100"),
        },
        {
            "name": "Low Win Rate, Very High Ratio",
            "win_rate": 0.40,
            "avg_win": Decimal("300"),
            "avg_loss": Decimal("100"),
        },
    ]

    for scenario in scenarios:
        result = sizer.calculate_position_size(
            portfolio_value=Decimal("100000"),
            price=Decimal("50.00"),
            win_rate=scenario["win_rate"],
            avg_win=scenario["avg_win"],
            avg_loss=scenario["avg_loss"],
        )

        print(f"{scenario['name']}:")
        print(f"  Shares: {result.shares}")
        print(f"  Position: ${result.dollar_amount:,.2f} ({result.portfolio_pct:.1%})")
        print(f"  Kelly: {result.kelly_full:.2%} -> {result.kelly_adjusted:.2%}")
        print()


if __name__ == "__main__":
    example_basic_position_sizing()
    example_aggressive_sizing()
    example_conservative_sizing()
    example_losing_strategy()
    example_multiple_scenarios()
