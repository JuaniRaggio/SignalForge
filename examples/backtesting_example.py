"""Example of using the backtesting engine to validate trading signals.

This example demonstrates how to:
1. Create price data and trading signals
2. Configure the backtest engine
3. Run a backtest
4. Analyze results and metrics
5. Log results to MLflow (optional)
"""

from datetime import datetime, timedelta

import polars as pl

from signalforge.ml.backtesting import BacktestConfig, BacktestEngine


def create_sample_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create sample price data and trading signals.

    Returns:
        Tuple of (price_df, signal_df)
    """
    # Create 100 days of price data with upward trend
    n_days = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    prices_data = []
    base_price = 100.0

    for i in range(n_days):
        close = base_price + i * 0.5 + (i % 7) * 0.3
        open_price = close - 0.2
        high = close + 0.5
        low = close - 0.5
        volume = 1000000 + (i % 10) * 100000

        prices_data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    prices = pl.DataFrame(prices_data)

    # Create simple trading signals: buy-hold-sell pattern
    signals_data = [{"timestamp": date, "signal": 0} for date in dates]

    # Buy at day 10, hold until day 50, then sell
    signals_data[10]["signal"] = 1  # Buy
    signals_data[50]["signal"] = -1  # Sell

    # Buy again at day 60, sell at day 90
    signals_data[60]["signal"] = 1  # Buy
    signals_data[90]["signal"] = -1  # Sell

    signals = pl.DataFrame(signals_data)

    return prices, signals


def main() -> None:
    """Run backtest example."""
    print("SignalForge Backtesting Engine Example")
    print("=" * 50)

    # Create sample data
    print("\n1. Creating sample price data and trading signals...")
    prices, signals = create_sample_data()
    print(f"   Price data: {prices.height} rows")
    print(f"   Signal data: {signals.height} rows")
    print(f"   Buy signals: {(signals['signal'] == 1).sum()}")
    print(f"   Sell signals: {(signals['signal'] == -1).sum()}")

    # Configure backtest
    print("\n2. Configuring backtest engine...")
    config = BacktestConfig(
        initial_capital=100000.0,  # $100k starting capital
        commission_pct=0.001,  # 0.1% commission
        slippage_pct=0.0005,  # 0.05% slippage
        position_size_pct=1.0,  # Use 100% of capital per trade
        allow_short=False,  # Long-only strategy
    )
    print(f"   Initial capital: ${config.initial_capital:,.2f}")
    print(f"   Commission: {config.commission_pct * 100:.2f}%")
    print(f"   Slippage: {config.slippage_pct * 100:.3f}%")
    print(f"   Position size: {config.position_size_pct * 100:.0f}% of capital")

    # Create engine and run backtest
    print("\n3. Running backtest...")
    engine = BacktestEngine(config)
    result = engine.run(prices, signals)

    # Display results
    print("\n4. Backtest Results:")
    print("=" * 50)
    print("\nPerformance Metrics:")
    print(f"  Total Return:       {result.metrics.total_return:>10.2f}%")
    print(f"  Annualized Return:  {result.metrics.annualized_return:>10.2f}%")
    print(f"  Sharpe Ratio:       {result.metrics.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:       {result.metrics.max_drawdown:>10.2f}%")
    print(f"  Volatility:         {result.metrics.volatility:>10.2f}%")

    print("\nTrade Statistics:")
    print(f"  Total Trades:       {result.metrics.total_trades:>10}")
    print(f"  Win Rate:           {result.metrics.win_rate:>10.2f}%")
    print(f"  Profit Factor:      {result.metrics.profit_factor:>10.2f}")
    print(f"  Avg Trade Return:   {result.metrics.avg_trade_return:>10.2f}%")

    # Display individual trades
    print("\n5. Individual Trades:")
    print("=" * 50)
    for i, trade in enumerate(result.trades, 1):
        print(f"\nTrade {i}:")
        print(f"  Entry:  {trade.entry_date.strftime('%Y-%m-%d')} @ ${trade.entry_price:.2f}")
        print(f"  Exit:   {trade.exit_date.strftime('%Y-%m-%d')} @ ${trade.exit_price:.2f}")
        print(f"  Size:   {trade.position_size:.2f} shares")
        print(f"  P&L:    ${trade.pnl:,.2f}")
        print(f"  Return: {trade.return_pct:.2f}%")

    # Display equity curve summary
    print("\n6. Equity Curve Summary:")
    print("=" * 50)
    initial_equity = float(result.equity_curve["equity"][0])
    final_equity = float(result.equity_curve["equity"][-1])
    max_equity = float(result.equity_curve["equity"].max())
    min_equity = float(result.equity_curve["equity"].min())

    print(f"  Initial Equity: ${initial_equity:,.2f}")
    print(f"  Final Equity:   ${final_equity:,.2f}")
    print(f"  Peak Equity:    ${max_equity:,.2f}")
    print(f"  Lowest Equity:  ${min_equity:,.2f}")

    print("\n" + "=" * 50)
    print("Backtest completed successfully!")

    # Optional: Log to MLflow
    # Uncomment the following to log results to MLflow:
    #
    # import mlflow
    #
    # with mlflow.start_run():
    #     mlflow.log_param("strategy", "simple_buy_hold")
    #     engine.log_to_mlflow(result)
    #     print("\nResults logged to MLflow!")


if __name__ == "__main__":
    main()
