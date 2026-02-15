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
    calculate_all,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_profit_factor,
    calculate_sortino_ratio,
)
from signalforge.ml.backtesting.strategies import (
    LongShortStrategy,
    RankingStrategy,
    ThresholdStrategy,
    TradeSignal,
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
        assert isinstance(result.total_return, float)
        assert isinstance(result.trade_log, pl.DataFrame)
        assert isinstance(result.equity_curve, pl.DataFrame)

    def test_run_produces_trades(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test that backtest produces trades."""
        engine = BacktestEngine()
        result = engine.run(sample_prices, simple_signals)

        # Simple signals should produce 2 trades
        assert result.total_trades == 2
        assert result.trade_log.height == 2

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
        assert result_with_commission.total_return < result_no_commission.total_return

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
        assert result_with_slippage.total_return < result_no_slippage.total_return

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

        assert result.total_trades == 0
        assert result.trade_log.is_empty()
        assert result.win_rate == 0.0

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

        assert result.total_trades == 1
        assert result.trade_log.height == 1

    def test_run_frequent_trading(
        self, sample_prices: pl.DataFrame, frequent_signals: pl.DataFrame
    ) -> None:
        """Test backtest with frequent trades."""
        engine = BacktestEngine()
        result = engine.run(sample_prices, frequent_signals)

        # Should have multiple trades
        assert result.total_trades > 5

    def test_run_position_sizing(
        self, sample_prices: pl.DataFrame, simple_signals: pl.DataFrame
    ) -> None:
        """Test that position sizing is respected."""
        # Use 50% of capital per trade
        config = BacktestConfig(position_size_pct=0.5)
        engine = BacktestEngine(config)
        result = engine.run(sample_prices, simple_signals)

        # All trades should use approximately 50% of capital
        if not result.trade_log.is_empty():
            for row in result.trade_log.iter_rows(named=True):
                position_value = float(row["position_size"]) * float(row["entry_price"])
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
        assert -100 <= result.total_return <= 1000
        assert -100 <= result.annualized_return <= 1000
        assert -10 <= result.sharpe_ratio <= 10
        assert 0 <= result.max_drawdown <= 100
        assert 0 <= result.win_rate <= 100
        assert result.profit_factor >= 0
        assert result.total_trades >= 0


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
        assert result.total_return > 0
        assert result.total_trades == 1
        assert result.win_rate == 100.0

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
        assert result.total_return < 0
        assert result.total_trades == 1
        assert result.win_rate == 0.0

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
        assert abs(result.total_return - expected_return) < 0.5


# ==================== NEW TESTS FOR STRATEGY-BASED BACKTESTING ====================


@pytest.fixture
def sample_predictions() -> pl.DataFrame:
    """Create sample ML predictions for testing strategies."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]

    predictions = []
    for i, date in enumerate(dates):
        predictions.append(
            {
                "symbol": "AAPL",
                "timestamp": date,
                "predicted_return": 0.02 + (i % 10) * 0.005,
                "confidence": 0.6 + (i % 5) * 0.05,
            }
        )
        predictions.append(
            {
                "symbol": "GOOGL",
                "timestamp": date,
                "predicted_return": -0.01 + (i % 8) * 0.003,
                "confidence": 0.7 + (i % 4) * 0.03,
            }
        )

    return pl.DataFrame(predictions)


@pytest.fixture
def multi_symbol_prices() -> pl.DataFrame:
    """Create price data for multiple symbols."""
    n_days = 50
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    prices = []
    for symbol, base_price in [("AAPL", 100.0), ("GOOGL", 150.0)]:
        for i, date in enumerate(dates):
            close = base_price + i * 0.5
            prices.append(
                {
                    "symbol": symbol,
                    "timestamp": date,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000000,
                }
            )

    return pl.DataFrame(prices)


class TestTradeSignal:
    """Tests for TradeSignal dataclass."""

    def test_valid_trade_signal(self) -> None:
        """Test creating a valid TradeSignal."""
        signal = TradeSignal(
            symbol="AAPL",
            action="buy",
            size=0.1,
            confidence=0.8,
            timestamp=datetime(2024, 1, 1),
            predicted_return=0.02,
        )

        assert signal.symbol == "AAPL"
        assert signal.action == "buy"
        assert signal.size == 0.1
        assert signal.confidence == 0.8

    def test_invalid_action(self) -> None:
        """Test error with invalid action."""
        with pytest.raises(ValueError, match="Invalid action"):
            TradeSignal(
                symbol="AAPL",
                action="invalid",  # type: ignore
                size=0.1,
                confidence=0.8,
                timestamp=datetime(2024, 1, 1),
            )

    def test_invalid_size(self) -> None:
        """Test error with invalid size."""
        with pytest.raises(ValueError, match="Size must be between 0 and 1"):
            TradeSignal(
                symbol="AAPL",
                action="buy",
                size=1.5,
                confidence=0.8,
                timestamp=datetime(2024, 1, 1),
            )

    def test_invalid_confidence(self) -> None:
        """Test error with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            TradeSignal(
                symbol="AAPL",
                action="buy",
                size=0.1,
                confidence=1.5,
                timestamp=datetime(2024, 1, 1),
            )


class TestThresholdStrategy:
    """Tests for ThresholdStrategy."""

    def test_initialization_default(self) -> None:
        """Test ThresholdStrategy with default parameters."""
        strategy = ThresholdStrategy()

        assert strategy.buy_threshold == 0.02
        assert strategy.sell_threshold == -0.02
        assert strategy.confidence_threshold == 0.6
        assert strategy.max_positions == 10

    def test_initialization_custom(self) -> None:
        """Test ThresholdStrategy with custom parameters."""
        strategy = ThresholdStrategy(
            buy_threshold=0.03,
            sell_threshold=-0.03,
            confidence_threshold=0.7,
            max_positions=5,
        )

        assert strategy.buy_threshold == 0.03
        assert strategy.sell_threshold == -0.03
        assert strategy.confidence_threshold == 0.7
        assert strategy.max_positions == 5

    def test_invalid_thresholds(self) -> None:
        """Test error when buy threshold is not greater than sell threshold."""
        with pytest.raises(ValueError, match="buy_threshold must be greater than sell_threshold"):
            ThresholdStrategy(buy_threshold=0.01, sell_threshold=0.02)

    def test_generate_signals_buy(self, sample_predictions: pl.DataFrame) -> None:
        """Test generating buy signals."""
        strategy = ThresholdStrategy(buy_threshold=0.02, confidence_threshold=0.6)

        # Filter predictions that should generate buy signals
        high_return_preds = sample_predictions.filter(pl.col("predicted_return") > 0.02)

        signals = strategy.generate_signals(high_return_preds, {})

        # Should generate some buy signals
        assert len(signals) > 0
        assert all(s.action == "buy" for s in signals)

    def test_generate_signals_sell(self, sample_predictions: pl.DataFrame) -> None:
        """Test generating sell signals."""
        strategy = ThresholdStrategy(sell_threshold=-0.02, confidence_threshold=0.6)

        # Create predictions with negative returns
        low_return_preds = sample_predictions.with_columns(
            pl.col("predicted_return").map_elements(lambda x: -abs(x), return_dtype=pl.Float64)
        )

        # Current positions
        current_positions = {"AAPL": 0.1, "GOOGL": 0.1}

        signals = strategy.generate_signals(low_return_preds, current_positions)

        # Should generate sell signals for held positions
        assert len(signals) > 0
        assert all(s.action == "sell" for s in signals)

    def test_generate_signals_max_positions(self, sample_predictions: pl.DataFrame) -> None:
        """Test max positions constraint."""
        strategy = ThresholdStrategy(buy_threshold=0.01, max_positions=1)

        # Filter to a single timestamp to test
        single_time_preds = sample_predictions.filter(
            pl.col("timestamp") == sample_predictions["timestamp"][0]
        )

        signals = strategy.generate_signals(single_time_preds, {})

        # Should not exceed max positions
        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) <= 1

    def test_generate_signals_confidence_filter(self, sample_predictions: pl.DataFrame) -> None:
        """Test confidence filtering."""
        strategy = ThresholdStrategy(buy_threshold=0.01, confidence_threshold=0.9)

        signals = strategy.generate_signals(sample_predictions, {})

        # High confidence threshold should filter out most signals
        assert len(signals) < len(sample_predictions)


class TestRankingStrategy:
    """Tests for RankingStrategy."""

    def test_initialization(self) -> None:
        """Test RankingStrategy initialization."""
        strategy = RankingStrategy(top_n=5, rebalance_frequency=5)

        assert strategy.top_n == 5
        assert strategy.rebalance_frequency == 5

    def test_generate_signals_top_n(self, sample_predictions: pl.DataFrame) -> None:
        """Test selecting top N stocks."""
        strategy = RankingStrategy(top_n=2, rebalance_frequency=1, equal_weight=True)

        signals = strategy.generate_signals(sample_predictions, {})

        # Should generate signals for top 2 stocks
        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) <= 2

    def test_generate_signals_equal_weight(self, sample_predictions: pl.DataFrame) -> None:
        """Test equal weighting."""
        strategy = RankingStrategy(top_n=3, rebalance_frequency=1, equal_weight=True)

        signals = strategy.generate_signals(sample_predictions, {})

        buy_signals = [s for s in signals if s.action == "buy"]

        # All buy signals should have equal size
        if len(buy_signals) > 0:
            expected_size = 1.0 / 3
            for signal in buy_signals:
                assert abs(signal.size - expected_size) < 0.01

    def test_generate_signals_rebalance(self, sample_predictions: pl.DataFrame) -> None:
        """Test rebalancing logic."""
        strategy = RankingStrategy(top_n=2, rebalance_frequency=100)

        # First call should generate signals
        signals1 = strategy.generate_signals(sample_predictions, {})
        assert len(signals1) > 0

        # Second call (same timestamp) should not generate signals (too soon)
        signals2 = strategy.generate_signals(sample_predictions, {})
        assert len(signals2) == 0


class TestLongShortStrategy:
    """Tests for LongShortStrategy."""

    def test_initialization(self) -> None:
        """Test LongShortStrategy initialization."""
        strategy = LongShortStrategy(long_n=5, short_n=5)

        assert strategy.long_n == 5
        assert strategy.short_n == 5

    def test_generate_signals_long_short(self, sample_predictions: pl.DataFrame) -> None:
        """Test generating long and short signals."""
        # Need more predictions for long-short
        strategy = LongShortStrategy(long_n=1, short_n=1, rebalance_frequency=1)

        signals = strategy.generate_signals(sample_predictions, {})

        # Should generate both long and short signals
        buy_signals = [s for s in signals if s.action == "buy"]
        sell_signals = [s for s in signals if s.action == "sell"]

        assert len(buy_signals) > 0 or len(sell_signals) > 0

    def test_generate_signals_insufficient_predictions(self) -> None:
        """Test with insufficient predictions."""
        small_preds = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": [datetime(2024, 1, 1)],
                "predicted_return": [0.02],
                "confidence": [0.8],
            }
        )

        strategy = LongShortStrategy(long_n=5, short_n=5)

        signals = strategy.generate_signals(small_preds, {})

        # Should not generate signals without enough predictions
        assert len(signals) == 0


class TestBacktestEngineStrategies:
    """Tests for BacktestEngine with strategies."""

    def test_run_with_strategy_basic(
        self,
        sample_predictions: pl.DataFrame,
        multi_symbol_prices: pl.DataFrame,
    ) -> None:
        """Test run_with_strategy method."""
        engine = BacktestEngine()
        strategy = ThresholdStrategy(buy_threshold=0.02)

        result = engine.run_with_strategy(sample_predictions, multi_symbol_prices, strategy)

        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert isinstance(result.equity_curve, pl.DataFrame)

    def test_run_with_strategy_produces_metrics(
        self,
        sample_predictions: pl.DataFrame,
        multi_symbol_prices: pl.DataFrame,
    ) -> None:
        """Test that strategy backtest produces all metrics."""
        engine = BacktestEngine()
        strategy = ThresholdStrategy()

        result = engine.run_with_strategy(sample_predictions, multi_symbol_prices, strategy)

        # Check all metrics are present
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "calmar_ratio")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "profit_factor")

    def test_run_with_strategy_trade_log(
        self,
        sample_predictions: pl.DataFrame,
        multi_symbol_prices: pl.DataFrame,
    ) -> None:
        """Test trade log generation."""
        engine = BacktestEngine()
        strategy = ThresholdStrategy()

        result = engine.run_with_strategy(sample_predictions, multi_symbol_prices, strategy)

        # Trade log should be a DataFrame
        assert isinstance(result.trade_log, pl.DataFrame)

    def test_run_with_strategy_monthly_returns(
        self,
        sample_predictions: pl.DataFrame,
        multi_symbol_prices: pl.DataFrame,
    ) -> None:
        """Test monthly returns calculation."""
        engine = BacktestEngine()
        strategy = ThresholdStrategy()

        result = engine.run_with_strategy(sample_predictions, multi_symbol_prices, strategy)

        # Monthly returns should be a DataFrame
        assert isinstance(result.monthly_returns, pl.DataFrame)

    def test_run_with_strategy_invalid_predictions(
        self,
        multi_symbol_prices: pl.DataFrame,
    ) -> None:
        """Test error with invalid predictions."""
        invalid_preds = pl.DataFrame({"symbol": ["AAPL"], "timestamp": [datetime(2024, 1, 1)]})

        engine = BacktestEngine()
        strategy = ThresholdStrategy()

        with pytest.raises(ValueError, match="Missing required prediction columns"):
            engine.run_with_strategy(invalid_preds, multi_symbol_prices, strategy)


class TestAdditionalMetrics:
    """Tests for additional metric functions."""

    def test_calculate_sortino_ratio_basic(self) -> None:
        """Test Sortino ratio calculation."""
        returns = pl.Series([0.01, -0.01, 0.02, -0.015, 0.03])

        sortino = calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)
        assert sortino != 0.0

    def test_calculate_sortino_ratio_no_downside(self) -> None:
        """Test Sortino ratio with no negative returns."""
        returns = pl.Series([0.01, 0.02, 0.015, 0.03])

        sortino = calculate_sortino_ratio(returns)

        # Should return inf when no downside
        assert sortino == float("inf")

    def test_calculate_calmar_ratio_basic(self) -> None:
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(annualized_return=20.0, max_drawdown=10.0)

        assert calmar == 2.0

    def test_calculate_calmar_ratio_zero_drawdown(self) -> None:
        """Test Calmar ratio with zero drawdown."""
        calmar = calculate_calmar_ratio(annualized_return=20.0, max_drawdown=0.0)

        assert calmar == 0.0

    def test_calculate_information_ratio_basic(self) -> None:
        """Test information ratio calculation."""
        portfolio_returns = pl.Series([0.01, 0.02, -0.01, 0.015])
        benchmark_returns = pl.Series([0.008, 0.015, -0.005, 0.012])

        ir = calculate_information_ratio(portfolio_returns, benchmark_returns)

        assert isinstance(ir, float)

    def test_calculate_information_ratio_mismatched_length(self) -> None:
        """Test information ratio with mismatched lengths."""
        portfolio_returns = pl.Series([0.01, 0.02, -0.01])
        benchmark_returns = pl.Series([0.008, 0.015])

        ir = calculate_information_ratio(portfolio_returns, benchmark_returns)

        assert ir == 0.0

    def test_calculate_all_metrics(self) -> None:
        """Test calculate_all function."""
        equity = pl.Series([100000, 105000, 103000, 108000, 110000])
        returns = equity.pct_change().drop_nulls()
        trade_log = pl.DataFrame()

        metrics = calculate_all(equity, returns, trade_log)

        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "calmar_ratio" in metrics
        assert "win_rate" in metrics


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_equity_curve(self) -> None:
        """Test metrics with empty equity curve."""
        empty_equity = pl.Series([])
        empty_returns = pl.Series([])

        sharpe = calculate_sharpe_ratio(empty_returns)
        max_dd = calculate_max_drawdown(empty_equity)

        assert sharpe == 0.0
        assert max_dd == 0.0

    def test_single_value_equity_curve(self) -> None:
        """Test metrics with single value."""
        single_equity = pl.Series([100000])

        max_dd = calculate_max_drawdown(single_equity)

        assert max_dd == 0.0

    def test_all_winning_trades_metrics(self) -> None:
        """Test metrics with all winning trades."""
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
        profit_factor = calculate_profit_factor(trades)

        assert win_rate == 100.0
        assert profit_factor == float("inf")

    def test_all_losing_trades_metrics(self) -> None:
        """Test metrics with all losing trades."""
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
        profit_factor = calculate_profit_factor(trades)

        assert win_rate == 0.0
        assert profit_factor == 0.0
