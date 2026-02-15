"""Example usage of the Concentration Alerts module."""

from decimal import Decimal

import polars as pl

from signalforge.risk.concentration import (
    AlertSeverity,
    AlertType,
    ConcentrationAnalyzer,
    ConcentrationConfig,
    Position,
)


def example_basic_analysis() -> None:
    """Example 1: Basic portfolio concentration analysis."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Portfolio Concentration Analysis")
    print("=" * 80)

    # Create analyzer with default configuration
    analyzer = ConcentrationAnalyzer()

    # Define a concentrated portfolio
    positions = [
        Position(symbol="AAPL", value=Decimal("60000"), weight=0.60),
        Position(symbol="MSFT", value=Decimal("40000"), weight=0.40),
    ]

    # Analyze portfolio
    result = analyzer.analyze_portfolio(positions, sector_map={})

    # Display results
    print(f"\nPortfolio HHI: {result.hhi_index:.3f}")
    print(f"Is Concentrated: {result.is_concentrated}")
    print(f"Largest Position: {result.largest_position_pct:.1%}")
    print(f"\nAlerts Generated: {len(result.alerts)}")

    for alert in result.alerts:
        print(f"\n[{alert.severity.value.upper()}] {alert.alert_type.value}")
        print(f"  {alert.message}")
        print(f"  Value: {alert.value:.1%}, Threshold: {alert.threshold:.1%}")


def example_sector_concentration() -> None:
    """Example 2: Sector concentration analysis."""
    print("\n" + "=" * 80)
    print("Example 2: Sector Concentration Analysis")
    print("=" * 80)

    # Create analyzer
    analyzer = ConcentrationAnalyzer()

    # Portfolio with sector concentration
    positions = [
        Position(symbol="AAPL", value=Decimal("40000"), weight=0.40, sector="Technology"),
        Position(symbol="MSFT", value=Decimal("30000"), weight=0.30, sector="Technology"),
        Position(symbol="GOOGL", value=Decimal("20000"), weight=0.20, sector="Technology"),
        Position(symbol="JPM", value=Decimal("10000"), weight=0.10, sector="Financials"),
    ]

    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "JPM": "Financials",
    }

    result = analyzer.analyze_portfolio(positions, sector_map)

    print(f"\nPortfolio HHI: {result.hhi_index:.3f}")
    print(f"Largest Sector: {result.largest_sector_pct:.1%}")
    print(f"\nSector Concentration Alerts:")

    sector_alerts = result.get_alerts_by_type(AlertType.SECTOR_CONCENTRATION)
    for alert in sector_alerts:
        print(f"  - {alert.message}")


def example_custom_thresholds() -> None:
    """Example 3: Custom concentration thresholds."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Concentration Thresholds")
    print("=" * 80)

    # Create custom configuration
    config = ConcentrationConfig(
        max_single_position_pct=0.25,  # 25% max per position
        max_sector_pct=0.40,  # 40% max per sector
        hhi_warning_threshold=0.20,
        hhi_critical_threshold=0.30,
    )

    analyzer = ConcentrationAnalyzer(config)

    positions = [
        Position(symbol="AAPL", value=Decimal("30000"), weight=0.30),
        Position(symbol="MSFT", value=Decimal("30000"), weight=0.30),
        Position(symbol="JPM", value=Decimal("20000"), weight=0.20),
        Position(symbol="BAC", value=Decimal("20000"), weight=0.20),
    ]

    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JPM": "Financials",
        "BAC": "Financials",
    }

    result = analyzer.analyze_portfolio(positions, sector_map)

    print(f"\nConfiguration:")
    print(f"  Max Single Position: {config.max_single_position_pct:.1%}")
    print(f"  Max Sector: {config.max_sector_pct:.1%}")
    print(f"  HHI Warning: {config.hhi_warning_threshold:.2f}")
    print(f"  HHI Critical: {config.hhi_critical_threshold:.2f}")

    print(f"\nResults:")
    print(f"  Portfolio HHI: {result.hhi_index:.3f}")
    print(f"  Total Alerts: {len(result.alerts)}")

    warnings = result.get_alerts_by_severity(AlertSeverity.WARNING)
    criticals = result.get_alerts_by_severity(AlertSeverity.CRITICAL)

    print(f"  Warning Alerts: {len(warnings)}")
    print(f"  Critical Alerts: {len(criticals)}")


def example_correlated_groups() -> None:
    """Example 4: Correlated asset group detection."""
    print("\n" + "=" * 80)
    print("Example 4: Correlated Asset Group Detection")
    print("=" * 80)

    analyzer = ConcentrationAnalyzer()

    positions = [
        Position(symbol="AAPL", value=Decimal("30000"), weight=0.30),
        Position(symbol="MSFT", value=Decimal("25000"), weight=0.25),
        Position(symbol="GOOGL", value=Decimal("20000"), weight=0.20),
        Position(symbol="JPM", value=Decimal("25000"), weight=0.25),
    ]

    # Correlation matrix showing high correlation between tech stocks
    correlation_matrix = pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "JPM"],
            "AAPL": [1.0, 0.85, 0.80, 0.20],
            "MSFT": [0.85, 1.0, 0.75, 0.15],
            "GOOGL": [0.80, 0.75, 1.0, 0.18],
            "JPM": [0.20, 0.15, 0.18, 1.0],
        }
    )

    result = analyzer.analyze_portfolio(positions, {}, correlation_matrix)

    print(f"\nPortfolio HHI: {result.hhi_index:.3f}")

    corr_alerts = result.get_alerts_by_type(AlertType.CORRELATED_GROUP)
    if corr_alerts:
        print(f"\nCorrelated Group Alerts:")
        for alert in corr_alerts:
            print(f"  - {alert.message}")
            print(f"    Group Size: {alert.metadata.get('group_size', 'N/A')} positions")
            print(f"    Total Weight: {alert.value:.1%}")
    else:
        print("\nNo correlated group alerts (groups are within limits)")


def example_well_diversified() -> None:
    """Example 5: Well-diversified portfolio."""
    print("\n" + "=" * 80)
    print("Example 5: Well-Diversified Portfolio")
    print("=" * 80)

    analyzer = ConcentrationAnalyzer()

    # Create a well-diversified portfolio with 20 equal positions
    positions = [
        Position(symbol=f"STOCK{i:02d}", value=Decimal("5000"), weight=0.05) for i in range(20)
    ]

    # Assign different sectors
    sectors = ["Technology", "Financials", "Healthcare", "Energy", "Consumer"]
    sector_map = {f"STOCK{i:02d}": sectors[i % len(sectors)] for i in range(20)}

    result = analyzer.analyze_portfolio(positions, sector_map)

    print(f"\nPortfolio Statistics:")
    print(f"  Number of Positions: {result.metadata.get('num_positions', 0)}")
    print(f"  Number of Sectors: {result.metadata.get('num_sectors', 0)}")
    print(f"  Portfolio HHI: {result.hhi_index:.3f}")
    print(f"  Largest Position: {result.largest_position_pct:.1%}")
    print(f"  Largest Sector: {result.largest_sector_pct:.1%}")
    print(f"\nIs Concentrated: {result.is_concentrated}")
    print(f"Total Alerts: {len(result.alerts)}")

    if result.alerts:
        print("\nAlerts:")
        for alert in result.alerts:
            print(f"  - [{alert.severity.value}] {alert.message}")
    else:
        print("\nNo concentration alerts - Portfolio is well diversified!")


def main() -> None:
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CONCENTRATION ALERTS MODULE EXAMPLES" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")

    example_basic_analysis()
    example_sector_concentration()
    example_custom_thresholds()
    example_correlated_groups()
    example_well_diversified()

    print("\n" + "=" * 80)
    print("Examples Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
