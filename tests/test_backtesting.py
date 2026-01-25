"""Comprehensive tests for backtesting engine and metrics.

This module tests all aspects of the backtesting functionality including:
- Configuration validation
- Trade execution simulation
- Performance metrics calculation
- Edge cases and error handling
- Integration with realistic trading scenarios
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import polars as pl
import pytest

from signalforge.ml.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestMetrics,
    BacktestResult,
    Trade,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
)
from signalforge.ml.backtesting.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_profit_factor,
)


@pytest.fixture
def sample_prices() -> pl.DataFrame:
    """Create sample price data for testing.

    Returns 100 days of price data with an upward trend.
    """
    n_days = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Generate realistic OHLCV data
    base_price = 100.0
    prices = []
    for i in range(n_days):
        close = base_price + i * 0.5 + (i % 7) * 0.3
        open_price = close - 0.2
        high = close + 0.5
        low = close - 0.5
        volume = 1000000 + (i % 10) * 100000

        prices.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pl.DataFrame(prices)


@pytest.fixture
def simple_signals() -> pl.DataFrame:
    """Create simple trading signals.

    Returns signals for buy-hold-sell pattern.
    """
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

    # Simple strategy: buy at day 10, sell at day 50, buy at day 60, sell at day 90
    signals = [0] * 100
    signals[10] = 1  # Buy
    signals[50] = -1  # Sell
    signals[60] = 1  # Buy
    signals[90] = -1  # Sell

    return pl.DataFrame({"timestamp": dates, "signal": signals})


@pytest.fixture
def frequent_signals() -> pl.DataFrame:
    """Create frequent trading signals for testing multiple trades."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

    # Frequent trading: buy every 10 days, sell 5 days later
    signals = [0] * 100
    for i in range(0, 100, 10):
        if i < 95:
            signals[i] = 1  # Buy
            if i + 5 < 100:
                signals[i + 5] = -1  # Sell

    return pl.DataFrame({"timestamp": dates, "signal": signals})


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test BacktestConfig with default values."""
        config = BacktestConfig()

        assert config.initial_capital == 100000.0
        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.0005
        assert config.position_size_pct == 1.0
        assert config.allow_short is False

    def test_custom_initialization(self) -> None:
        """Test BacktestConfig with custom values."""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission_pct=0.002,
            slippage_pct=0.001,
            position_size_pct=0.5,
            allow_short=True,
        )

        assert config.initial_capital == 50000.0
        assert config.commission_pct == 0.002
        assert config.slippage_pct == 0.001
        assert config.position_size_pct == 0.5
        assert config.allow_short is True

    def test_invalid_initial_capital(self) -> None:
        """Test error with invalid initial capital."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            BacktestConfig(initial_capital=0.0)

        with pytest.raises(ValueError, match="initial_capital must be positive"):
            BacktestConfig(initial_capital=-1000.0)

    def test_invalid_commission(self) -> None:
        """Test error with negative commission."""
        with pytest.raises(ValueError, match="commission_pct cannot be negative"):
            BacktestConfig(commission_pct=-0.01)

    def test_invalid_slippage(self) -> None:
        """Test error with negative slippage."""
        with pytest.raises(ValueError, match="slippage_pct cannot be negative"):
            BacktestConfig(slippage_pct=-0.01)

    def test_invalid_position_size(self) -> None:
        """Test error with invalid position size."""
        with pytest.raises(ValueError, match="position_size_pct must be between 0 and 1"):
            BacktestConfig(position_size_pct=0.0)

        with pytest.raises(ValueError, match="position_size_pct must be between 0 and 1"):
            BacktestConfig(position_size_pct=1.5)

    def test_total_transaction_cost_property(self) -> None:
        """Test total transaction cost calculation."""
        config = BacktestConfig(commission_pct=0.001, slippage_pct=0.0005)

        assert config.total_transaction_cost_pct == 0.0015


class TestTrade:
    """Tests for Trade dataclass."""

    def test_valid_trade(self) -> None:
        """Test creating a valid Trade."""
        trade = Trade(
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 10),
            entry_price=100.0,
            exit_price=105.0,
            position_size=100.0,
            direction="long",
            pnl=500.0,
            return_pct=5.0,
        )

        assert trade.entry_price == 100.0
        assert trade.exit_price == 105.0
        assert trade.direction == "long"
        assert trade.pnl == 500.0

    def test_invalid_entry_price(self) -> None:
        """Test error with invalid entry price."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=0.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            )

    def test_invalid_exit_price(self) -> None:
        """Test error with invalid exit price."""
        with pytest.raises(ValueError, match="exit_price must be positive"):
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=100.0,
                exit_price=-5.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            )

    def test_invalid_position_size(self) -> None:
        """Test error with invalid position size."""
        with pytest.raises(ValueError, match="position_size must be positive"):
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=100.0,
                exit_price=105.0,
                position_size=0.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            )

    def test_invalid_direction(self) -> None:
        """Test error with invalid direction."""
        with pytest.raises(ValueError, match="direction must be"):
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 10),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="invalid",  # type: ignore
                pnl=500.0,
                return_pct=5.0,
            )

    def test_invalid_dates(self) -> None:
        """Test error when exit date is before entry date."""
        with pytest.raises(ValueError, match="exit_date must be after entry_date"):
            Trade(
                entry_date=datetime(2024, 1, 10),
                exit_date=datetime(2024, 1, 1),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            )


class TestBacktestMetrics:
    """Tests for BacktestMetrics dataclass."""

    def test_valid_metrics(self) -> None:
        """Test creating valid BacktestMetrics."""
        metrics = BacktestMetrics(
            total_return=15.5,
            annualized_return=12.3,
            sharpe_ratio=1.8,
            max_drawdown=8.5,
            win_rate=65.0,
            profit_factor=2.1,
            total_trades=50,
            avg_trade_return=0.31,
            volatility=18.2,
        )

        assert metrics.total_return == 15.5
        assert metrics.sharpe_ratio == 1.8
        assert metrics.total_trades == 50

    def test_invalid_total_trades(self) -> None:
        """Test error with negative total trades."""
        with pytest.raises(ValueError, match="total_trades cannot be negative"):
            BacktestMetrics(
                total_return=15.5,
                annualized_return=12.3,
                sharpe_ratio=1.8,
                max_drawdown=8.5,
                win_rate=65.0,
                profit_factor=2.1,
                total_trades=-5,
                avg_trade_return=0.31,
                volatility=18.2,
            )

    def test_invalid_win_rate(self) -> None:
        """Test error with invalid win rate."""
        with pytest.raises(ValueError, match="win_rate must be between 0 and 100"):
            BacktestMetrics(
                total_return=15.5,
                annualized_return=12.3,
                sharpe_ratio=1.8,
                max_drawdown=8.5,
                win_rate=150.0,
                profit_factor=2.1,
                total_trades=50,
                avg_trade_return=0.31,
                volatility=18.2,
            )

    def test_invalid_profit_factor(self) -> None:
        """Test error with negative profit factor."""
        with pytest.raises(ValueError, match="profit_factor cannot be negative"):
            BacktestMetrics(
                total_return=15.5,
                annualized_return=12.3,
                sharpe_ratio=1.8,
                max_drawdown=8.5,
                win_rate=65.0,
                profit_factor=-1.0,
                total_trades=50,
                avg_trade_return=0.31,
                volatility=18.2,
            )

    def test_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = BacktestMetrics(
            total_return=15.5,
            annualized_return=12.3,
            sharpe_ratio=1.8,
            max_drawdown=8.5,
            win_rate=65.0,
            profit_factor=2.1,
            total_trades=50,
            avg_trade_return=0.31,
            volatility=18.2,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["total_return"] == 15.5
        assert metrics_dict["sharpe_ratio"] == 1.8
        assert metrics_dict["total_trades"] == 50
        assert len(metrics_dict) == 9


class TestMetricFunctions:
    """Tests for individual metric calculation functions."""

    def test_calculate_sharpe_ratio_basic(self) -> None:
        """Test Sharpe ratio calculation with basic returns."""
        returns = pl.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.03, -0.002])

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

        assert isinstance(sharpe, float)
        assert sharpe != 0.0

    def test_calculate_sharpe_ratio_with_risk_free_rate(self) -> None:
        """Test Sharpe ratio with non-zero risk-free rate."""
        returns = pl.Series([0.01, 0.02, 0.015, 0.018, 0.012])

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        assert isinstance(sharpe, float)

    def test_calculate_sharpe_ratio_empty(self) -> None:
        """Test Sharpe ratio with empty series."""
        returns = pl.Series([])

        sharpe = calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_zero_volatility(self) -> None:
        """Test Sharpe ratio when volatility is zero."""
        returns = pl.Series([0.01, 0.01, 0.01, 0.01])

        sharpe = calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_calculate_max_drawdown_basic(self) -> None:
        """Test max drawdown calculation."""
        equity = pl.Series([100000, 105000, 103000, 108000, 102000, 110000, 95000, 112000])

        max_dd = calculate_max_drawdown(equity)

        assert max_dd > 0.0
        # Should be approximately 13.6% (from 110000 to 95000)
        assert 13.0 < max_dd < 14.0

    def test_calculate_max_drawdown_no_drawdown(self) -> None:
        """Test max drawdown with monotonically increasing equity."""
        equity = pl.Series([100000, 105000, 110000, 115000, 120000])

        max_dd = calculate_max_drawdown(equity)

        assert max_dd == 0.0

    def test_calculate_max_drawdown_empty(self) -> None:
        """Test max drawdown with empty series."""
        equity = pl.Series([])

        max_dd = calculate_max_drawdown(equity)

        assert max_dd == 0.0

    def test_calculate_win_rate_basic(self) -> None:
        """Test win rate calculation."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 10),
                exit_date=datetime(2024, 1, 15),
                entry_price=100.0,
                exit_price=98.0,
                position_size=100.0,
                direction="long",
                pnl=-200.0,
                return_pct=-2.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 20),
                exit_date=datetime(2024, 1, 25),
                entry_price=100.0,
                exit_price=103.0,
                position_size=100.0,
                direction="long",
                pnl=300.0,
                return_pct=3.0,
            ),
        ]

        win_rate = calculate_win_rate(trades)

        # 2 out of 3 trades are winners
        assert abs(win_rate - 66.67) < 0.1

    def test_calculate_win_rate_all_winners(self) -> None:
        """Test win rate with all winning trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, i),
                exit_date=datetime(2024, 1, i + 1),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            )
            for i in range(1, 6)
        ]

        win_rate = calculate_win_rate(trades)

        assert win_rate == 100.0

    def test_calculate_win_rate_all_losers(self) -> None:
        """Test win rate with all losing trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, i),
                exit_date=datetime(2024, 1, i + 1),
                entry_price=100.0,
                exit_price=95.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-5.0,
            )
            for i in range(1, 6)
        ]

        win_rate = calculate_win_rate(trades)

        assert win_rate == 0.0

    def test_calculate_win_rate_no_trades(self) -> None:
        """Test win rate with no trades."""
        trades = []

        win_rate = calculate_win_rate(trades)

        assert win_rate == 0.0

    def test_calculate_profit_factor_basic(self) -> None:
        """Test profit factor calculation."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 10),
                exit_date=datetime(2024, 1, 15),
                entry_price=100.0,
                exit_price=98.0,
                position_size=100.0,
                direction="long",
                pnl=-200.0,
                return_pct=-2.0,
            ),
        ]

        profit_factor = calculate_profit_factor(trades)

        # Gross profit = 500, Gross loss = 200, PF = 500/200 = 2.5
        assert abs(profit_factor - 2.5) < 0.01

    def test_calculate_profit_factor_no_losses(self) -> None:
        """Test profit factor with only winning trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, i),
                exit_date=datetime(2024, 1, i + 1),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            )
            for i in range(1, 4)
        ]

        profit_factor = calculate_profit_factor(trades)

        assert profit_factor == float("inf")

    def test_calculate_profit_factor_no_profits(self) -> None:
        """Test profit factor with only losing trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, i),
                exit_date=datetime(2024, 1, i + 1),
                entry_price=100.0,
                exit_price=95.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-5.0,
            )
            for i in range(1, 4)
        ]

        profit_factor = calculate_profit_factor(trades)

        assert profit_factor == 0.0

    def test_calculate_annualized_return_basic(self) -> None:
        """Test annualized return calculation."""
        # 20% return over 504 days (2 years)
        annualized = calculate_annualized_return(20.0, 504)

        # Should be approximately 9.5% annualized
        assert 9.0 < annualized < 10.0

    def test_calculate_annualized_return_short_period(self) -> None:
        """Test annualized return for period less than 1 year."""
        # For periods less than a year, should return total return
        annualized = calculate_annualized_return(10.0, 100)

        assert annualized == 10.0

    def test_calculate_annualized_volatility_basic(self) -> None:
        """Test annualized volatility calculation."""
        returns = pl.Series([0.01, -0.01, 0.02, -0.015, 0.03, -0.02])

        volatility = calculate_annualized_volatility(returns)

        assert volatility > 0.0


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_initialization_default(self) -> None:
        """Test engine initialization with defaults."""
        engine = BacktestEngine()

        assert engine.config.initial_capital == 100000.0
        assert engine.config.commission_pct == 0.001

    def test_initialization_custom_config(self) -> None:
        """Test engine initialization with custom config."""
        config = BacktestConfig(initial_capital=50000.0, commission_pct=0.002)
        engine = BacktestEngine(config)

        assert engine.config.initial_capital == 50000.0
        assert engine.config.commission_pct == 0.002

    def test_run_basic(self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame) -> None:
        """Test basic backtest run."""
        engine = BacktestEngine()
        result = engine.run(sample_prices, simple_signals)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, BacktestMetrics)
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, pl.DataFrame)

    def test_run_produces_trades(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test that backtest produces trades."""
        engine = BacktestEngine()
        result = engine.run(sample_prices, simple_signals)

        # Simple signals should produce 2 trades
        assert result.metrics.total_trades == 2
        assert len(result.trades) == 2

    def test_run_equity_curve(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test equity curve generation."""
        engine = BacktestEngine()
        result = engine.run(sample_prices, simple_signals)

        assert "timestamp" in result.equity_curve.columns
        assert "equity" in result.equity_curve.columns
        assert result.equity_curve.height == sample_prices.height

    def test_run_with_commission(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test that commissions are properly applied."""
        # Run with no commission
        config_no_commission = BacktestConfig(commission_pct=0.0, slippage_pct=0.0)
        engine_no_commission = BacktestEngine(config_no_commission)
        result_no_commission = engine_no_commission.run(sample_prices, simple_signals)

        # Run with commission
        config_with_commission = BacktestConfig(commission_pct=0.01, slippage_pct=0.0)
        engine_with_commission = BacktestEngine(config_with_commission)
        result_with_commission = engine_with_commission.run(sample_prices, simple_signals)

        # Results with commission should have lower returns
        assert (
            result_with_commission.metrics.total_return < result_no_commission.metrics.total_return
        )

    def test_run_with_slippage(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test that slippage is properly applied."""
        # Run with no slippage
        config_no_slippage = BacktestConfig(commission_pct=0.0, slippage_pct=0.0)
        engine_no_slippage = BacktestEngine(config_no_slippage)
        result_no_slippage = engine_no_slippage.run(sample_prices, simple_signals)

        # Run with slippage
        config_with_slippage = BacktestConfig(commission_pct=0.0, slippage_pct=0.01)
        engine_with_slippage = BacktestEngine(config_with_slippage)
        result_with_slippage = engine_with_slippage.run(sample_prices, simple_signals)

        # Results with slippage should have lower returns
        assert result_with_slippage.metrics.total_return < result_no_slippage.metrics.total_return

    def test_run_empty_prices(self, simple_signals: pl.DataFrame) -> None:
        """Test error with empty price DataFrame."""
        empty_prices = pl.DataFrame(
            {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        )

        engine = BacktestEngine()

        with pytest.raises(ValueError, match="Price DataFrame cannot be empty"):
            engine.run(empty_prices, simple_signals)

    def test_run_empty_signals(self, sample_prices: pl.DataFrame) -> None:
        """Test error with empty signal DataFrame."""
        empty_signals = pl.DataFrame({"timestamp": [], "signal": []})

        engine = BacktestEngine()

        with pytest.raises(ValueError, match="Signal DataFrame cannot be empty"):
            engine.run(sample_prices, empty_signals)

    def test_run_missing_price_columns(self, simple_signals: pl.DataFrame) -> None:
        """Test error with missing price columns."""
        invalid_prices = pl.DataFrame(
            {"timestamp": [datetime(2024, 1, 1)], "close": [100.0]}  # Missing other columns
        )

        engine = BacktestEngine()

        with pytest.raises(ValueError, match="Missing required price columns"):
            engine.run(invalid_prices, simple_signals)

    def test_run_missing_signal_columns(self, sample_prices: pl.DataFrame) -> None:
        """Test error with missing signal columns."""
        invalid_signals = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)]})  # Missing signal

        engine = BacktestEngine()

        with pytest.raises(ValueError, match="Missing required signal columns"):
            engine.run(sample_prices, invalid_signals)

    def test_run_invalid_signal_values(self, sample_prices: pl.DataFrame) -> None:
        """Test error with invalid signal values."""
        invalid_signals = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
                "signal": [2, 3, 1, -1, 5, 0, 1, -1, 0, 1],  # Contains invalid values
            }
        )

        engine = BacktestEngine()

        with pytest.raises(ValueError, match="Invalid signal values"):
            engine.run(sample_prices, invalid_signals)

    def test_run_no_trades(self, sample_prices: pl.DataFrame) -> None:
        """Test backtest with signals that produce no trades."""
        # All hold signals
        no_trade_signals = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)],
                "signal": [0] * 100,
            }
        )

        engine = BacktestEngine()
        result = engine.run(sample_prices, no_trade_signals)

        assert result.metrics.total_trades == 0
        assert len(result.trades) == 0
        assert result.metrics.win_rate == 0.0

    def test_run_one_trade(self, sample_prices: pl.DataFrame) -> None:
        """Test backtest with exactly one trade."""
        one_trade_signals = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)],
                "signal": [1 if i == 10 else -1 if i == 50 else 0 for i in range(100)],
            }
        )

        engine = BacktestEngine()
        result = engine.run(sample_prices, one_trade_signals)

        assert result.metrics.total_trades == 1
        assert len(result.trades) == 1

    def test_run_frequent_trading(
        self, sample_prices: pl.DataFrame, frequent_signals: pl.DataFrame
    ) -> None:
        """Test backtest with frequent trades."""
        engine = BacktestEngine()
        result = engine.run(sample_prices, frequent_signals)

        # Should have multiple trades
        assert result.metrics.total_trades > 5

    def test_run_position_sizing(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test that position sizing is respected."""
        # Use 50% of capital per trade
        config = BacktestConfig(position_size_pct=0.5)
        engine = BacktestEngine(config)
        result = engine.run(sample_prices, simple_signals)

        # All trades should use approximately 50% of capital
        for trade in result.trades:
            position_value = trade.position_size * trade.entry_price
            # Should be around 50000 (50% of 100000)
            assert 45000 < position_value < 55000

    def test_run_no_matching_timestamps(self, sample_prices: pl.DataFrame) -> None:
        """Test error when prices and signals have no matching timestamps."""
        non_matching_signals = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1) + timedelta(days=i) for i in range(10)],
                "signal": [1, -1, 0, 1, -1, 0, 1, -1, 0, 0],
            }
        )

        engine = BacktestEngine()

        with pytest.raises(ValueError, match="No matching timestamps"):
            engine.run(sample_prices, non_matching_signals)

    def test_log_to_mlflow(self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame) -> None:
        """Test logging results to MLflow."""
        with (
            patch("mlflow.log_param") as mock_log_param,
            patch("mlflow.log_metric") as mock_log_metric,
        ):
            engine = BacktestEngine()
            result = engine.run(sample_prices, simple_signals)

            engine.log_to_mlflow(result)

            # Verify MLflow functions were called
            assert mock_log_param.called
            assert mock_log_metric.called

    def test_metrics_are_reasonable(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test that calculated metrics are reasonable."""
        engine = BacktestEngine()
        result = engine.run(sample_prices, simple_signals)

        # Check metrics are within reasonable ranges
        assert -100 <= result.metrics.total_return <= 1000
        assert -100 <= result.metrics.annualized_return <= 1000
        assert -10 <= result.metrics.sharpe_ratio <= 10
        assert 0 <= result.metrics.max_drawdown <= 100
        assert 0 <= result.metrics.win_rate <= 100
        assert result.metrics.profit_factor >= 0
        assert result.metrics.total_trades >= 0
        assert result.metrics.volatility >= 0


class TestBacktestIntegration:
    """Integration tests for complete backtesting workflows."""

    def test_profitable_strategy(self, sample_prices: pl.DataFrame) -> None:
        """Test a strategy that should be profitable."""
        # Simple trend-following: buy when price crosses above SMA, sell when below
        # For our upward-trending data, this should be profitable

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        signals = [0] * 100

        # Buy early, sell late (capture the uptrend)
        signals[5] = 1
        signals[95] = -1

        strategy_signals = pl.DataFrame({"timestamp": dates, "signal": signals})

        engine = BacktestEngine()
        result = engine.run(sample_prices, strategy_signals)

        # Should have positive return
        assert result.metrics.total_return > 0
        assert result.metrics.total_trades == 1
        assert result.metrics.win_rate == 100.0

    def test_losing_strategy(self, sample_prices: pl.DataFrame) -> None:
        """Test a strategy that should lose money."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        signals = [0] * 100

        # Buy at peak, sell shortly after - the upward trend is too strong
        # so we need high transaction costs to guarantee a loss
        signals[80] = 1
        signals[82] = -1

        strategy_signals = pl.DataFrame({"timestamp": dates, "signal": signals})

        # Use high transaction costs to ensure loss on short hold period
        config = BacktestConfig(commission_pct=0.01, slippage_pct=0.01)
        engine = BacktestEngine(config)
        result = engine.run(sample_prices, strategy_signals)

        # With high transaction costs and short hold, should lose money
        assert result.metrics.total_return < 0
        assert result.metrics.total_trades == 1
        assert result.metrics.win_rate == 0.0

    def test_buy_and_hold_equivalent(self, sample_prices: pl.DataFrame) -> None:
        """Test buy-and-hold strategy."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        signals = [0] * 100

        # Buy at start, sell at end
        signals[0] = 1
        signals[99] = -1

        buy_hold_signals = pl.DataFrame({"timestamp": dates, "signal": signals})

        config = BacktestConfig(commission_pct=0.0, slippage_pct=0.0)
        engine = BacktestEngine(config)
        result = engine.run(sample_prices, buy_hold_signals)

        # Calculate expected return
        initial_price = float(sample_prices["close"][0])
        final_price = float(sample_prices["close"][-1])
        expected_return = ((final_price - initial_price) / initial_price) * 100.0

        # Should match buy-and-hold return (within tolerance for rounding)
        assert abs(result.metrics.total_return - expected_return) < 0.5
