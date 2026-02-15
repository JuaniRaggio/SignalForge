"""Comprehensive tests for performance metrics calculation.

This test suite covers all aspects of the PerformanceCalculator including:
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Trade statistics
- CAGR and volatility calculations
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import polars as pl
import pytest

from signalforge.benchmark.performance import (
    PerformanceCalculator,
    PerformanceMetrics,
    annualize_returns,
    annualize_volatility,
)
from signalforge.ml.backtesting.engine import Trade


class TestAnnualizeFunctions:
    """Test annualization helper functions."""

    def test_annualize_returns_daily(self) -> None:
        """Test annualizing daily returns."""
        daily_return = 0.001  # 0.1% per day
        annualized = annualize_returns(daily_return, 252)
        assert annualized == pytest.approx(0.252, rel=1e-5)

    def test_annualize_returns_monthly(self) -> None:
        """Test annualizing monthly returns."""
        monthly_return = 0.02  # 2% per month
        annualized = annualize_returns(monthly_return, 12)
        assert annualized == pytest.approx(0.24, rel=1e-5)

    def test_annualize_returns_zero(self) -> None:
        """Test annualizing zero returns."""
        annualized = annualize_returns(0.0, 252)
        assert annualized == 0.0

    def test_annualize_returns_negative(self) -> None:
        """Test annualizing negative returns."""
        daily_return = -0.001
        annualized = annualize_returns(daily_return, 252)
        assert annualized == pytest.approx(-0.252, rel=1e-5)

    def test_annualize_returns_invalid_periods(self) -> None:
        """Test that invalid periods raises ValueError."""
        with pytest.raises(ValueError, match="periods must be positive"):
            annualize_returns(0.1, 0)

        with pytest.raises(ValueError, match="periods must be positive"):
            annualize_returns(0.1, -252)

    def test_annualize_volatility_daily(self) -> None:
        """Test annualizing daily volatility."""
        daily_vol = 0.02  # 2% per day
        annualized = annualize_volatility(daily_vol, 252)
        assert annualized == pytest.approx(0.02 * (252**0.5), rel=1e-5)

    def test_annualize_volatility_monthly(self) -> None:
        """Test annualizing monthly volatility."""
        monthly_vol = 0.05  # 5% per month
        annualized = annualize_volatility(monthly_vol, 12)
        assert annualized == pytest.approx(0.05 * (12**0.5), rel=1e-5)

    def test_annualize_volatility_zero(self) -> None:
        """Test annualizing zero volatility."""
        annualized = annualize_volatility(0.0, 252)
        assert annualized == 0.0

    def test_annualize_volatility_invalid_periods(self) -> None:
        """Test that invalid periods raises ValueError."""
        with pytest.raises(ValueError, match="periods must be positive"):
            annualize_volatility(0.02, 0)

        with pytest.raises(ValueError, match="periods must be positive"):
            annualize_volatility(0.02, -252)


class TestPerformanceCalculatorInit:
    """Test PerformanceCalculator initialization."""

    def test_default_initialization(self) -> None:
        """Test default calculator initialization."""
        calc = PerformanceCalculator()
        assert calc.risk_free_rate == 0.0
        assert calc.periods_per_year == 252

    def test_custom_initialization(self) -> None:
        """Test custom calculator initialization."""
        calc = PerformanceCalculator(risk_free_rate=0.02, periods_per_year=365)
        assert calc.risk_free_rate == 0.02
        assert calc.periods_per_year == 365

    def test_initialization_negative_risk_free_rate(self) -> None:
        """Test that negative risk-free rate raises ValueError."""
        with pytest.raises(ValueError, match="risk_free_rate cannot be negative"):
            PerformanceCalculator(risk_free_rate=-0.01)

    def test_initialization_invalid_periods(self) -> None:
        """Test that invalid periods raises ValueError."""
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            PerformanceCalculator(periods_per_year=0)

        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            PerformanceCalculator(periods_per_year=-252)


class TestCalculateReturns:
    """Test calculate_returns method."""

    def test_calculate_returns_simple(self) -> None:
        """Test calculating returns from simple equity curve."""
        equity = [Decimal("100"), Decimal("105"), Decimal("110")]
        calc = PerformanceCalculator()
        returns = calc.calculate_returns(equity)

        assert returns.len() == 2
        assert returns[0] == pytest.approx(0.05, rel=1e-5)
        assert returns[1] == pytest.approx(0.047619, rel=1e-4)

    def test_calculate_returns_flat(self) -> None:
        """Test calculating returns from flat equity curve."""
        equity = [Decimal("100"), Decimal("100"), Decimal("100")]
        calc = PerformanceCalculator()
        returns = calc.calculate_returns(equity)

        assert returns.len() == 2
        assert all(r == 0.0 for r in returns)

    def test_calculate_returns_declining(self) -> None:
        """Test calculating returns from declining equity curve."""
        equity = [Decimal("100"), Decimal("95"), Decimal("90")]
        calc = PerformanceCalculator()
        returns = calc.calculate_returns(equity)

        assert returns.len() == 2
        assert returns[0] == pytest.approx(-0.05, rel=1e-5)
        assert returns[1] == pytest.approx(-0.052632, rel=1e-4)

    def test_calculate_returns_empty_list(self) -> None:
        """Test that empty equity curve raises ValueError."""
        calc = PerformanceCalculator()
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            calc.calculate_returns([])

    def test_calculate_returns_negative_values(self) -> None:
        """Test that negative equity values raise ValueError."""
        equity = [Decimal("100"), Decimal("-50")]
        calc = PerformanceCalculator()
        with pytest.raises(ValueError, match="equity_curve values must be positive"):
            calc.calculate_returns(equity)

    def test_calculate_returns_zero_values(self) -> None:
        """Test that zero equity values raise ValueError."""
        equity = [Decimal("100"), Decimal("0")]
        calc = PerformanceCalculator()
        with pytest.raises(ValueError, match="equity_curve values must be positive"):
            calc.calculate_returns(equity)


class TestCalculateSharpeRatio:
    """Test calculate_sharpe_ratio method."""

    def test_sharpe_ratio_positive_returns(self) -> None:
        """Test Sharpe ratio with positive returns."""
        returns = pl.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        calc = PerformanceCalculator(risk_free_rate=0.0, periods_per_year=252)
        sharpe = calc.calculate_sharpe_ratio(returns)

        # Expected: mean ~0.015, std ~0.00447, Sharpe ~47.6
        assert sharpe > 0
        assert sharpe == pytest.approx(47.6, rel=0.1)

    def test_sharpe_ratio_with_risk_free_rate(self) -> None:
        """Test Sharpe ratio with non-zero risk-free rate."""
        returns = pl.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        calc = PerformanceCalculator(risk_free_rate=0.02, periods_per_year=252)
        sharpe = calc.calculate_sharpe_ratio(returns)

        # With risk-free rate, Sharpe should be lower
        assert sharpe > 0

    def test_sharpe_ratio_negative_returns(self) -> None:
        """Test Sharpe ratio with negative returns."""
        returns = pl.Series([-0.01, -0.02, -0.015, -0.01, -0.02])
        calc = PerformanceCalculator(risk_free_rate=0.0, periods_per_year=252)
        sharpe = calc.calculate_sharpe_ratio(returns)

        # Should be negative
        assert sharpe < 0

    def test_sharpe_ratio_zero_volatility(self) -> None:
        """Test Sharpe ratio with zero volatility (constant returns)."""
        returns = pl.Series([0.01, 0.01, 0.01, 0.01])
        calc = PerformanceCalculator()
        sharpe = calc.calculate_sharpe_ratio(returns)

        # Zero volatility should return 0
        assert sharpe == 0.0

    def test_sharpe_ratio_empty_series(self) -> None:
        """Test Sharpe ratio with empty series."""
        returns = pl.Series([])
        calc = PerformanceCalculator()
        sharpe = calc.calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_sharpe_ratio_single_value(self) -> None:
        """Test Sharpe ratio with single value."""
        returns = pl.Series([0.01])
        calc = PerformanceCalculator()
        sharpe = calc.calculate_sharpe_ratio(returns)

        assert sharpe == 0.0


class TestCalculateSortinoRatio:
    """Test calculate_sortino_ratio method."""

    def test_sortino_ratio_positive_returns(self) -> None:
        """Test Sortino ratio with mostly positive returns."""
        returns = pl.Series([0.01, 0.02, -0.005, 0.015, 0.02])
        calc = PerformanceCalculator(risk_free_rate=0.0, periods_per_year=252)
        sortino = calc.calculate_sortino_ratio(returns)

        # Sortino should be higher than Sharpe due to only penalizing downside
        assert sortino > 0

    def test_sortino_ratio_no_downside(self) -> None:
        """Test Sortino ratio with no downside returns."""
        returns = pl.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        calc = PerformanceCalculator(risk_free_rate=0.0, periods_per_year=252)
        sortino = calc.calculate_sortino_ratio(returns)

        # Should return a high value (100.0)
        assert sortino == pytest.approx(100.0)

    def test_sortino_ratio_all_negative(self) -> None:
        """Test Sortino ratio with all negative returns."""
        returns = pl.Series([-0.01, -0.02, -0.015, -0.01, -0.02])
        calc = PerformanceCalculator(risk_free_rate=0.0, periods_per_year=252)
        sortino = calc.calculate_sortino_ratio(returns)

        # Should be negative
        assert sortino < 0

    def test_sortino_ratio_with_risk_free_rate(self) -> None:
        """Test Sortino ratio with non-zero risk-free rate."""
        returns = pl.Series([0.01, 0.02, -0.005, 0.015, 0.02])
        calc = PerformanceCalculator(risk_free_rate=0.02, periods_per_year=252)
        sortino = calc.calculate_sortino_ratio(returns)

        assert sortino != 0.0

    def test_sortino_ratio_empty_series(self) -> None:
        """Test Sortino ratio with empty series."""
        returns = pl.Series([])
        calc = PerformanceCalculator()
        sortino = calc.calculate_sortino_ratio(returns)

        assert sortino == 0.0


class TestCalculateCalmarRatio:
    """Test calculate_calmar_ratio method."""

    def test_calmar_ratio_positive(self) -> None:
        """Test Calmar ratio with positive returns and drawdown."""
        returns = pl.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        calc = PerformanceCalculator(periods_per_year=252)
        calmar = calc.calculate_calmar_ratio(returns, 0.15)

        # Expected: annualized return / max_dd
        assert calmar > 0

    def test_calmar_ratio_zero_drawdown(self) -> None:
        """Test Calmar ratio with zero drawdown."""
        returns = pl.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        calc = PerformanceCalculator()
        calmar = calc.calculate_calmar_ratio(returns, 0.0)

        # Zero drawdown should return 0
        assert calmar == 0.0

    def test_calmar_ratio_negative_returns(self) -> None:
        """Test Calmar ratio with negative returns."""
        returns = pl.Series([-0.01, -0.02, -0.015, -0.01, -0.02])
        calc = PerformanceCalculator(periods_per_year=252)
        calmar = calc.calculate_calmar_ratio(returns, 0.15)

        # Should be negative
        assert calmar < 0

    def test_calmar_ratio_empty_series(self) -> None:
        """Test Calmar ratio with empty series."""
        returns = pl.Series([])
        calc = PerformanceCalculator()
        calmar = calc.calculate_calmar_ratio(returns, 0.15)

        assert calmar == 0.0


class TestCalculateMaxDrawdown:
    """Test calculate_max_drawdown method."""

    def test_max_drawdown_simple(self) -> None:
        """Test max drawdown with simple decline."""
        equity = [Decimal("100"), Decimal("110"), Decimal("90"), Decimal("95")]
        calc = PerformanceCalculator()
        max_dd, duration = calc.calculate_max_drawdown(equity)

        # Peak at 110, trough at 90: (110-90)/110 = 18.18%
        assert max_dd == pytest.approx(0.1818, rel=1e-3)
        assert duration > 0

    def test_max_drawdown_no_decline(self) -> None:
        """Test max drawdown with no decline."""
        equity = [Decimal("100"), Decimal("105"), Decimal("110"), Decimal("115")]
        calc = PerformanceCalculator()
        max_dd, duration = calc.calculate_max_drawdown(equity)

        # No drawdown
        assert max_dd == 0.0
        assert duration == 0

    def test_max_drawdown_full_recovery(self) -> None:
        """Test max drawdown with full recovery."""
        equity = [Decimal("100"), Decimal("110"), Decimal("90"), Decimal("110")]
        calc = PerformanceCalculator()
        max_dd, duration = calc.calculate_max_drawdown(equity)

        # Peak at 110, trough at 90, recovery at 110
        assert max_dd == pytest.approx(0.1818, rel=1e-3)
        assert duration == 2  # From peak (index 1) to recovery (index 3)

    def test_max_drawdown_no_recovery(self) -> None:
        """Test max drawdown with no recovery."""
        equity = [Decimal("100"), Decimal("110"), Decimal("90"), Decimal("85")]
        calc = PerformanceCalculator()
        max_dd, duration = calc.calculate_max_drawdown(equity)

        # Peak at 110, trough at 85: (110-85)/110 = 22.73%
        assert max_dd == pytest.approx(0.2273, rel=1e-3)
        # Duration from peak to end (no recovery)
        assert duration == 3

    def test_max_drawdown_multiple_drawdowns(self) -> None:
        """Test max drawdown with multiple drawdown periods."""
        equity = [
            Decimal("100"),
            Decimal("110"),
            Decimal("95"),
            Decimal("105"),
            Decimal("85"),
        ]
        calc = PerformanceCalculator()
        max_dd, duration = calc.calculate_max_drawdown(equity)

        # Max drawdown should be from 110 to 85
        assert max_dd == pytest.approx(0.2273, rel=1e-3)

    def test_max_drawdown_empty_list(self) -> None:
        """Test max drawdown with empty equity curve."""
        calc = PerformanceCalculator()
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            calc.calculate_max_drawdown([])

    def test_max_drawdown_single_value(self) -> None:
        """Test max drawdown with single value."""
        equity = [Decimal("100")]
        calc = PerformanceCalculator()
        max_dd, duration = calc.calculate_max_drawdown(equity)

        assert max_dd == 0.0
        assert duration == 0


class TestCalculateCAGR:
    """Test calculate_cagr method."""

    def test_cagr_one_year_positive(self) -> None:
        """Test CAGR with positive growth over one year."""
        calc = PerformanceCalculator()
        cagr = calc.calculate_cagr(Decimal("100000"), Decimal("115000"), 365)

        # 15% growth in one year
        assert cagr == pytest.approx(0.15, rel=1e-5)

    def test_cagr_multiple_years(self) -> None:
        """Test CAGR over multiple years."""
        calc = PerformanceCalculator()
        cagr = calc.calculate_cagr(Decimal("100000"), Decimal("150000"), 730)  # 2 years

        # (150000/100000)^(1/2) - 1 = 22.47%
        assert cagr == pytest.approx(0.2247, rel=1e-3)

    def test_cagr_negative_growth(self) -> None:
        """Test CAGR with negative growth."""
        calc = PerformanceCalculator()
        cagr = calc.calculate_cagr(Decimal("100000"), Decimal("85000"), 365)

        # -15% in one year
        assert cagr == pytest.approx(-0.15, rel=1e-5)

    def test_cagr_no_growth(self) -> None:
        """Test CAGR with no growth."""
        calc = PerformanceCalculator()
        cagr = calc.calculate_cagr(Decimal("100000"), Decimal("100000"), 365)

        assert cagr == pytest.approx(0.0, abs=1e-10)

    def test_cagr_less_than_year(self) -> None:
        """Test CAGR for period less than one year."""
        calc = PerformanceCalculator()
        cagr = calc.calculate_cagr(Decimal("100000"), Decimal("110000"), 180)  # ~6 months

        # 10% in 6 months -> ~21.55% annualized
        assert cagr > 0.21
        assert cagr < 0.22

    def test_cagr_zero_initial(self) -> None:
        """Test CAGR with zero initial value."""
        calc = PerformanceCalculator()
        cagr = calc.calculate_cagr(Decimal("0"), Decimal("100000"), 365)

        assert cagr == 0.0

    def test_cagr_zero_days(self) -> None:
        """Test CAGR with zero days."""
        calc = PerformanceCalculator()
        cagr = calc.calculate_cagr(Decimal("100000"), Decimal("110000"), 0)

        assert cagr == 0.0

    def test_cagr_negative_values(self) -> None:
        """Test CAGR with negative values raises ValueError."""
        calc = PerformanceCalculator()
        with pytest.raises(ValueError, match="initial and final values must be non-negative"):
            calc.calculate_cagr(Decimal("-100000"), Decimal("110000"), 365)

        with pytest.raises(ValueError, match="initial and final values must be non-negative"):
            calc.calculate_cagr(Decimal("100000"), Decimal("-110000"), 365)

    def test_cagr_negative_days(self) -> None:
        """Test CAGR with negative days raises ValueError."""
        calc = PerformanceCalculator()
        with pytest.raises(ValueError, match="days cannot be negative"):
            calc.calculate_cagr(Decimal("100000"), Decimal("110000"), -365)


class TestCalculateWinRate:
    """Test calculate_win_rate method."""

    def test_win_rate_all_winners(self) -> None:
        """Test win rate with all winning trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=105.0,
                exit_price=110.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=4.76,
            ),
        ]
        calc = PerformanceCalculator()
        win_rate = calc.calculate_win_rate(trades)

        assert win_rate == 1.0

    def test_win_rate_all_losers(self) -> None:
        """Test win rate with all losing trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=95.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=95.0,
                exit_price=90.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-5.26,
            ),
        ]
        calc = PerformanceCalculator()
        win_rate = calc.calculate_win_rate(trades)

        assert win_rate == 0.0

    def test_win_rate_mixed(self) -> None:
        """Test win rate with mixed trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=105.0,
                exit_price=100.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-4.76,
            ),
            Trade(
                entry_date=datetime(2024, 1, 5),
                exit_date=datetime(2024, 1, 6),
                entry_price=100.0,
                exit_price=110.0,
                position_size=100.0,
                direction="long",
                pnl=1000.0,
                return_pct=10.0,
            ),
        ]
        calc = PerformanceCalculator()
        win_rate = calc.calculate_win_rate(trades)

        # 2 winners out of 3
        assert win_rate == pytest.approx(0.6667, rel=1e-3)

    def test_win_rate_no_trades(self) -> None:
        """Test win rate with no trades."""
        calc = PerformanceCalculator()
        win_rate = calc.calculate_win_rate([])

        assert win_rate == 0.0


class TestCalculateProfitFactor:
    """Test calculate_profit_factor method."""

    def test_profit_factor_profitable(self) -> None:
        """Test profit factor with profitable trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=105.0,
                exit_price=100.0,
                position_size=100.0,
                direction="long",
                pnl=-300.0,
                return_pct=-4.76,
            ),
        ]
        calc = PerformanceCalculator()
        pf = calc.calculate_profit_factor(trades)

        # 500 / 300 = 1.667
        assert pf == pytest.approx(1.6667, rel=1e-3)

    def test_profit_factor_no_losses(self) -> None:
        """Test profit factor with no losing trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=105.0,
                exit_price=110.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=4.76,
            ),
        ]
        calc = PerformanceCalculator()
        pf = calc.calculate_profit_factor(trades)

        # Should return infinity
        assert pf == float("inf")

    def test_profit_factor_no_profits(self) -> None:
        """Test profit factor with no winning trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=95.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=95.0,
                exit_price=90.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-5.26,
            ),
        ]
        calc = PerformanceCalculator()
        pf = calc.calculate_profit_factor(trades)

        # No profits, should return 0
        assert pf == 0.0

    def test_profit_factor_no_trades(self) -> None:
        """Test profit factor with no trades."""
        calc = PerformanceCalculator()
        pf = calc.calculate_profit_factor([])

        assert pf == 0.0

    def test_profit_factor_break_even(self) -> None:
        """Test profit factor with break-even trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=100.0,
                position_size=100.0,
                direction="long",
                pnl=0.0,
                return_pct=0.0,
            ),
        ]
        calc = PerformanceCalculator()
        pf = calc.calculate_profit_factor(trades)

        # No profits or losses
        assert pf == 0.0


class TestCalculateExpectancy:
    """Test calculate_expectancy method."""

    def test_expectancy_positive(self) -> None:
        """Test expectancy with positive expected value."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=110.0,
                position_size=100.0,
                direction="long",
                pnl=1000.0,
                return_pct=10.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=110.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=-500.0,
                return_pct=-4.55,
            ),
        ]
        calc = PerformanceCalculator()
        expectancy = calc.calculate_expectancy(trades)

        # (0.5 * 1000) - (0.5 * 500) = 250
        assert expectancy == pytest.approx(250.0, rel=1e-5)

    def test_expectancy_negative(self) -> None:
        """Test expectancy with negative expected value."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=105.0,
                exit_price=95.0,
                position_size=100.0,
                direction="long",
                pnl=-1000.0,
                return_pct=-9.52,
            ),
        ]
        calc = PerformanceCalculator()
        expectancy = calc.calculate_expectancy(trades)

        # (0.5 * 500) - (0.5 * 1000) = -250
        assert expectancy == pytest.approx(-250.0, rel=1e-5)

    def test_expectancy_all_winners(self) -> None:
        """Test expectancy with all winning trades."""
        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=105.0,
                exit_price=110.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=4.76,
            ),
        ]
        calc = PerformanceCalculator()
        expectancy = calc.calculate_expectancy(trades)

        # (1.0 * 500) - (0.0 * 0) = 500
        assert expectancy == pytest.approx(500.0, rel=1e-5)

    def test_expectancy_no_trades(self) -> None:
        """Test expectancy with no trades."""
        calc = PerformanceCalculator()
        expectancy = calc.calculate_expectancy([])

        assert expectancy == 0.0


class TestCalculateAll:
    """Test calculate_all method for comprehensive metrics."""

    def test_calculate_all_comprehensive(self) -> None:
        """Test comprehensive metrics calculation with realistic data."""
        equity = [
            Decimal("100000"),
            Decimal("105000"),
            Decimal("103000"),
            Decimal("108000"),
            Decimal("110000"),
        ]

        trades = [
            Trade(
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 2),
                entry_price=100.0,
                exit_price=105.0,
                position_size=100.0,
                direction="long",
                pnl=500.0,
                return_pct=5.0,
            ),
            Trade(
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 4),
                entry_price=105.0,
                exit_price=103.0,
                position_size=100.0,
                direction="long",
                pnl=-200.0,
                return_pct=-1.9,
            ),
            Trade(
                entry_date=datetime(2024, 1, 5),
                exit_date=datetime(2024, 1, 6),
                entry_price=103.0,
                exit_price=110.0,
                position_size=100.0,
                direction="long",
                pnl=700.0,
                return_pct=6.8,
            ),
        ]

        calc = PerformanceCalculator(risk_free_rate=0.02, periods_per_year=252)
        metrics = calc.calculate_all(equity, trades)

        # Verify all metrics are present and valid
        assert metrics.sharpe_ratio != 0.0
        assert metrics.sortino_ratio != 0.0
        assert metrics.calmar_ratio != 0.0
        assert 0 <= metrics.max_drawdown <= 1
        assert metrics.max_drawdown_duration >= 0
        assert metrics.cagr != 0.0
        assert metrics.volatility >= 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.profit_factor > 0
        assert metrics.avg_win > 0
        assert metrics.avg_loss >= 0
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1

    def test_calculate_all_no_trades(self) -> None:
        """Test metrics calculation with no trades."""
        equity = [Decimal("100000"), Decimal("105000"), Decimal("110000")]

        calc = PerformanceCalculator()
        metrics = calc.calculate_all(equity, [])

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0
        assert metrics.expectancy == 0.0

    def test_calculate_all_flat_equity(self) -> None:
        """Test metrics with flat equity curve."""
        equity = [Decimal("100000"), Decimal("100000"), Decimal("100000")]

        calc = PerformanceCalculator()
        metrics = calc.calculate_all(equity, [])

        assert metrics.sharpe_ratio == 0.0  # Zero volatility
        assert metrics.max_drawdown == 0.0
        assert metrics.cagr == pytest.approx(0.0, abs=1e-10)
        assert metrics.volatility == 0.0

    def test_calculate_all_empty_equity(self) -> None:
        """Test that empty equity curve raises ValueError."""
        calc = PerformanceCalculator()
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            calc.calculate_all([], [])


class TestPerformanceMetricsValidation:
    """Test PerformanceMetrics dataclass validation."""

    def test_valid_metrics(self) -> None:
        """Test creating valid metrics."""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            max_drawdown=0.15,
            max_drawdown_duration=10,
            cagr=0.12,
            volatility=0.18,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=500.0,
            avg_loss=300.0,
            expectancy=100.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
        )

        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_trades == 10

    def test_invalid_total_trades(self) -> None:
        """Test that negative total trades raises ValueError."""
        with pytest.raises(ValueError, match="total_trades cannot be negative"):
            PerformanceMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                calmar_ratio=0.8,
                max_drawdown=0.15,
                max_drawdown_duration=10,
                cagr=0.12,
                volatility=0.18,
                win_rate=0.6,
                profit_factor=1.8,
                avg_win=500.0,
                avg_loss=300.0,
                expectancy=100.0,
                total_trades=-1,
                winning_trades=6,
                losing_trades=4,
            )

    def test_invalid_win_rate(self) -> None:
        """Test that invalid win rate raises ValueError."""
        with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
            PerformanceMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                calmar_ratio=0.8,
                max_drawdown=0.15,
                max_drawdown_duration=10,
                cagr=0.12,
                volatility=0.18,
                win_rate=1.5,  # Invalid
                profit_factor=1.8,
                avg_win=500.0,
                avg_loss=300.0,
                expectancy=100.0,
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
            )

    def test_invalid_profit_factor(self) -> None:
        """Test that negative profit factor raises ValueError."""
        with pytest.raises(ValueError, match="profit_factor cannot be negative"):
            PerformanceMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                calmar_ratio=0.8,
                max_drawdown=0.15,
                max_drawdown_duration=10,
                cagr=0.12,
                volatility=0.18,
                win_rate=0.6,
                profit_factor=-1.8,  # Invalid
                avg_win=500.0,
                avg_loss=300.0,
                expectancy=100.0,
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
            )

    def test_invalid_max_drawdown(self) -> None:
        """Test that negative max drawdown raises ValueError."""
        with pytest.raises(ValueError, match="max_drawdown cannot be negative"):
            PerformanceMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                calmar_ratio=0.8,
                max_drawdown=-0.15,  # Invalid
                max_drawdown_duration=10,
                cagr=0.12,
                volatility=0.18,
                win_rate=0.6,
                profit_factor=1.8,
                avg_win=500.0,
                avg_loss=300.0,
                expectancy=100.0,
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
            )

    def test_invalid_avg_loss(self) -> None:
        """Test that negative avg loss raises ValueError."""
        with pytest.raises(ValueError, match="avg_loss must be positive"):
            PerformanceMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                calmar_ratio=0.8,
                max_drawdown=0.15,
                max_drawdown_duration=10,
                cagr=0.12,
                volatility=0.18,
                win_rate=0.6,
                profit_factor=1.8,
                avg_win=500.0,
                avg_loss=-300.0,  # Invalid (should be positive absolute value)
                expectancy=100.0,
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
            )
