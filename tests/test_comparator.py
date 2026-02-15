"""Comprehensive tests for benchmark comparator module.

This test suite covers all functionality of the BenchmarkComparator class,
including edge cases, error conditions, and financial calculation accuracy.
"""

from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest

from signalforge.benchmark.comparator import (
    BenchmarkComparator,
    BenchmarkData,
    ComparisonResult,
)


class TestBenchmarkData:
    """Tests for BenchmarkData dataclass."""

    def test_valid_benchmark_data(self) -> None:
        """Test creating valid benchmark data."""
        returns = pl.Series([0.01, 0.02, -0.005])
        equity = [Decimal("100"), Decimal("101"), Decimal("103.02"), Decimal("102.5")]

        data = BenchmarkData(name="SPY", returns=returns, equity_curve=equity)

        assert data.name == "SPY"
        assert data.returns.len() == 3
        assert len(data.equity_curve) == 4

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises ValueError."""
        returns = pl.Series([0.01, 0.02])
        equity = [Decimal("100"), Decimal("101"), Decimal("103")]

        with pytest.raises(ValueError, match="name cannot be empty"):
            BenchmarkData(name="", returns=returns, equity_curve=equity)

    def test_empty_returns_raises_error(self) -> None:
        """Test that empty returns raises ValueError."""
        returns = pl.Series([], dtype=pl.Float64)
        equity = [Decimal("100"), Decimal("101")]

        with pytest.raises(ValueError, match="empty returns"):
            BenchmarkData(name="SPY", returns=returns, equity_curve=equity)

    def test_short_equity_curve_raises_error(self) -> None:
        """Test that equity curve with less than 2 points raises error."""
        returns = pl.Series([0.01])
        equity = [Decimal("100")]

        with pytest.raises(ValueError, match="at least 2 points"):
            BenchmarkData(name="SPY", returns=returns, equity_curve=equity)

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched returns and equity lengths raises error."""
        returns = pl.Series([0.01, 0.02])
        equity = [Decimal("100"), Decimal("101")]  # Should be 3 elements

        with pytest.raises(ValueError, match="must equal equity_curve length - 1"):
            BenchmarkData(name="SPY", returns=returns, equity_curve=equity)


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_valid_comparison_result(self) -> None:
        """Test creating valid comparison result."""
        result = ComparisonResult(
            strategy_name="TestStrategy",
            benchmark_name="SPY",
            alpha=5.5,
            beta=1.2,
            correlation=0.85,
            tracking_error=3.5,
            information_ratio=1.57,
            up_capture=105.0,
            down_capture=95.0,
            relative_drawdown=2.5,
        )

        assert result.strategy_name == "TestStrategy"
        assert result.benchmark_name == "SPY"
        assert result.alpha == 5.5
        assert result.beta == 1.2

    def test_empty_strategy_name_raises_error(self) -> None:
        """Test that empty strategy name raises ValueError."""
        with pytest.raises(ValueError, match="strategy_name cannot be empty"):
            ComparisonResult(
                strategy_name="",
                benchmark_name="SPY",
                alpha=0.0,
                beta=1.0,
                correlation=0.5,
                tracking_error=2.0,
                information_ratio=0.0,
                up_capture=100.0,
                down_capture=100.0,
                relative_drawdown=0.0,
            )

    def test_empty_benchmark_name_raises_error(self) -> None:
        """Test that empty benchmark name raises ValueError."""
        with pytest.raises(ValueError, match="benchmark_name cannot be empty"):
            ComparisonResult(
                strategy_name="Strategy",
                benchmark_name="",
                alpha=0.0,
                beta=1.0,
                correlation=0.5,
                tracking_error=2.0,
                information_ratio=0.0,
                up_capture=100.0,
                down_capture=100.0,
                relative_drawdown=0.0,
            )

    def test_invalid_correlation_raises_error(self) -> None:
        """Test that correlation outside [-1, 1] raises ValueError."""
        with pytest.raises(ValueError, match="correlation must be in"):
            ComparisonResult(
                strategy_name="Strategy",
                benchmark_name="SPY",
                alpha=0.0,
                beta=1.0,
                correlation=1.5,  # Invalid
                tracking_error=2.0,
                information_ratio=0.0,
                up_capture=100.0,
                down_capture=100.0,
                relative_drawdown=0.0,
            )

    def test_negative_tracking_error_raises_error(self) -> None:
        """Test that negative tracking error raises ValueError."""
        with pytest.raises(ValueError, match="tracking_error cannot be negative"):
            ComparisonResult(
                strategy_name="Strategy",
                benchmark_name="SPY",
                alpha=0.0,
                beta=1.0,
                correlation=0.5,
                tracking_error=-2.0,  # Invalid
                information_ratio=0.0,
                up_capture=100.0,
                down_capture=100.0,
                relative_drawdown=0.0,
            )

    def test_to_dict_conversion(self) -> None:
        """Test conversion to dictionary."""
        result = ComparisonResult(
            strategy_name="TestStrategy",
            benchmark_name="SPY",
            alpha=5.5,
            beta=1.2,
            correlation=0.85,
            tracking_error=3.5,
            information_ratio=1.57,
            up_capture=105.0,
            down_capture=95.0,
            relative_drawdown=2.5,
        )

        result_dict = result.to_dict()

        assert result_dict["strategy_name"] == "TestStrategy"
        assert result_dict["benchmark_name"] == "SPY"
        assert result_dict["alpha"] == 5.5
        assert result_dict["beta"] == 1.2
        assert len(result_dict) == 10


class TestBenchmarkComparatorInit:
    """Tests for BenchmarkComparator initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization with zero risk-free rate."""
        comparator = BenchmarkComparator()

        assert comparator.risk_free_rate == 0.0
        assert len(comparator.benchmarks) == 0

    def test_custom_risk_free_rate(self) -> None:
        """Test initialization with custom risk-free rate."""
        comparator = BenchmarkComparator(risk_free_rate=0.02)

        assert comparator.risk_free_rate == 0.02

    def test_negative_risk_free_rate_raises_error(self) -> None:
        """Test that negative risk-free rate raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            BenchmarkComparator(risk_free_rate=-0.01)

    def test_unreasonably_high_risk_free_rate_raises_error(self) -> None:
        """Test that risk-free rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="unreasonably high"):
            BenchmarkComparator(risk_free_rate=1.5)


class TestAddBenchmark:
    """Tests for add_benchmark method."""

    def test_add_single_benchmark(self) -> None:
        """Test adding a single benchmark."""
        comparator = BenchmarkComparator()
        returns = pl.Series([0.01, 0.02])
        equity = [Decimal("100"), Decimal("101"), Decimal("103.02")]

        comparator.add_benchmark("SPY", returns, equity)

        assert "SPY" in comparator.benchmarks
        assert comparator.benchmarks["SPY"].name == "SPY"

    def test_add_multiple_benchmarks(self) -> None:
        """Test adding multiple benchmarks."""
        comparator = BenchmarkComparator()

        returns1 = pl.Series([0.01, 0.02])
        equity1 = [Decimal("100"), Decimal("101"), Decimal("103.02")]

        returns2 = pl.Series([0.015, 0.025])
        equity2 = [Decimal("100"), Decimal("101.5"), Decimal("104.04")]

        comparator.add_benchmark("SPY", returns1, equity1)
        comparator.add_benchmark("QQQ", returns2, equity2)

        assert len(comparator.benchmarks) == 2
        assert "SPY" in comparator.benchmarks
        assert "QQQ" in comparator.benchmarks

    def test_duplicate_benchmark_raises_error(self) -> None:
        """Test that adding duplicate benchmark raises ValueError."""
        comparator = BenchmarkComparator()
        returns = pl.Series([0.01, 0.02])
        equity = [Decimal("100"), Decimal("101"), Decimal("103.02")]

        comparator.add_benchmark("SPY", returns, equity)

        with pytest.raises(ValueError, match="already exists"):
            comparator.add_benchmark("SPY", returns, equity)


class TestCalculateBeta:
    """Tests for calculate_beta method."""

    def test_beta_perfect_correlation(self) -> None:
        """Test beta with perfectly correlated returns."""
        comparator = BenchmarkComparator()
        # For beta = 2, strategy returns should be 2 * benchmark returns (centered)
        # Beta = Cov(S, B) / Var(B)
        # If S = 2*B, then Cov(S,B) = 2*Var(B), so Beta = 2
        benchmark = pl.Series([0.00, 0.01, -0.01, 0.02, -0.02])
        strategy = pl.Series([0.00, 0.02, -0.02, 0.04, -0.04])  # 2x benchmark

        beta = comparator.calculate_beta(strategy, benchmark)

        # Beta should be 2.0 (strategy is 2x benchmark)
        assert abs(beta - 2.0) < 0.01

    def test_beta_same_volatility(self) -> None:
        """Test beta with same volatility returns."""
        comparator = BenchmarkComparator()
        # Identical returns should give beta of 1.0
        benchmark = pl.Series([0.00, 0.01, -0.01, 0.02, -0.02, 0.015])
        strategy = pl.Series([0.00, 0.01, -0.01, 0.02, -0.02, 0.015])  # Same as benchmark

        beta = comparator.calculate_beta(strategy, benchmark)

        # Beta should be 1.0 (identical returns)
        assert abs(beta - 1.0) < 0.01

    def test_beta_lower_volatility(self) -> None:
        """Test beta with lower volatility returns."""
        comparator = BenchmarkComparator()
        # Strategy has half the volatility of benchmark
        benchmark = pl.Series([0.00, 0.02, -0.02, 0.04, -0.04])
        strategy = pl.Series([0.00, 0.01, -0.01, 0.02, -0.02])  # 0.5x benchmark

        beta = comparator.calculate_beta(strategy, benchmark)

        # Beta should be 0.5 (strategy is half as volatile)
        assert abs(beta - 0.5) < 0.01

    def test_beta_negative_correlation(self) -> None:
        """Test beta with negatively correlated returns."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.02, -0.01, 0.03])
        benchmark = pl.Series([-0.02, 0.01, -0.03])

        beta = comparator.calculate_beta(strategy, benchmark)

        # Beta should be negative
        assert beta < 0

    def test_beta_zero_variance_benchmark(self) -> None:
        """Test beta when benchmark has zero variance."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.01, 0.01])  # No variance

        beta = comparator.calculate_beta(strategy, benchmark)

        assert beta == 0.0

    def test_beta_insufficient_data(self) -> None:
        """Test beta with insufficient data."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01])
        benchmark = pl.Series([0.01])

        beta = comparator.calculate_beta(strategy, benchmark)

        assert beta == 0.0

    def test_beta_empty_series(self) -> None:
        """Test beta with empty series."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([], dtype=pl.Float64)
        benchmark = pl.Series([], dtype=pl.Float64)

        beta = comparator.calculate_beta(strategy, benchmark)

        assert beta == 0.0

    def test_beta_mismatched_lengths(self) -> None:
        """Test beta with mismatched series lengths."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.02])

        beta = comparator.calculate_beta(strategy, benchmark)

        assert beta == 0.0

    def test_beta_with_nulls(self) -> None:
        """Test beta with null values in series."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, None, 0.03])
        benchmark = pl.Series([0.01, 0.02, 0.03])

        beta = comparator.calculate_beta(strategy, benchmark)

        # Should drop nulls and calculate
        assert beta >= 0.0


class TestCalculateAlpha:
    """Tests for calculate_alpha method."""

    def test_alpha_outperformance(self) -> None:
        """Test alpha calculation for outperforming strategy."""
        comparator = BenchmarkComparator(risk_free_rate=0.0)
        # Strategy consistently outperforms with similar volatility but higher returns
        import random
        random.seed(42)
        benchmark = pl.Series([random.gauss(0.0004, 0.01) for _ in range(252)])
        strategy = pl.Series([r + 0.0002 for r in benchmark])  # Add constant outperformance

        alpha = comparator.calculate_alpha(strategy, benchmark)

        # Alpha should be positive (outperformance)
        assert alpha > 0

    def test_alpha_underperformance(self) -> None:
        """Test alpha calculation for underperforming strategy."""
        comparator = BenchmarkComparator(risk_free_rate=0.0)
        # Strategy consistently underperforms with similar volatility but lower returns
        import random
        random.seed(44)
        benchmark = pl.Series([random.gauss(0.0004, 0.01) for _ in range(252)])
        strategy = pl.Series([r - 0.0002 for r in benchmark])  # Consistently lag

        alpha = comparator.calculate_alpha(strategy, benchmark)

        # Alpha should be negative (underperformance)
        assert alpha < 0

    def test_alpha_with_risk_free_rate(self) -> None:
        """Test alpha calculation with non-zero risk-free rate."""
        comparator = BenchmarkComparator(risk_free_rate=0.02)
        strategy = pl.Series([0.01] * 252)
        benchmark = pl.Series([0.01] * 252)

        alpha = comparator.calculate_alpha(strategy, benchmark)

        # Alpha should account for risk-free rate
        assert isinstance(alpha, float)

    def test_alpha_insufficient_data(self) -> None:
        """Test alpha with insufficient data."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01])
        benchmark = pl.Series([0.01])

        alpha = comparator.calculate_alpha(strategy, benchmark)

        assert alpha == 0.0

    def test_alpha_mismatched_lengths(self) -> None:
        """Test alpha with mismatched series lengths."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.02])

        alpha = comparator.calculate_alpha(strategy, benchmark)

        assert alpha == 0.0


class TestCalculateCorrelation:
    """Tests for calculate_correlation method."""

    def test_correlation_perfect_positive(self) -> None:
        """Test correlation with perfectly correlated returns."""
        comparator = BenchmarkComparator()
        # Perfectly proportional returns
        strategy = pl.Series([0.00, 0.01, 0.02, 0.03, 0.04])
        benchmark = pl.Series([0.00, 0.01, 0.02, 0.03, 0.04])

        corr = comparator.calculate_correlation(strategy, benchmark)

        assert abs(corr - 1.0) < 0.01

    def test_correlation_perfect_negative(self) -> None:
        """Test correlation with perfectly negatively correlated returns."""
        comparator = BenchmarkComparator()
        # Perfectly inversely proportional returns
        strategy = pl.Series([0.04, 0.03, 0.02, 0.01, 0.00])
        benchmark = pl.Series([0.00, 0.01, 0.02, 0.03, 0.04])

        corr = comparator.calculate_correlation(strategy, benchmark)

        assert abs(corr - (-1.0)) < 0.01

    def test_correlation_no_correlation(self) -> None:
        """Test correlation with uncorrelated returns."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, -0.01, 0.01, -0.01])
        benchmark = pl.Series([0.01, 0.01, -0.01, -0.01])

        corr = comparator.calculate_correlation(strategy, benchmark)

        # Should be close to zero
        assert abs(corr) < 0.5

    def test_correlation_zero_variance(self) -> None:
        """Test correlation when one series has zero variance."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.01, 0.01])

        corr = comparator.calculate_correlation(strategy, benchmark)

        assert corr == 0.0

    def test_correlation_clamped_to_range(self) -> None:
        """Test that correlation is clamped to [-1, 1]."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.02, 0.03])

        corr = comparator.calculate_correlation(strategy, benchmark)

        assert -1.0 <= corr <= 1.0


class TestCalculateTrackingError:
    """Tests for calculate_tracking_error method."""

    def test_tracking_error_identical_returns(self) -> None:
        """Test tracking error with identical returns."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03] * 100)
        benchmark = pl.Series([0.01, 0.02, 0.03] * 100)

        te = comparator.calculate_tracking_error(strategy, benchmark)

        # Should be near zero for identical returns
        assert te < 0.01

    def test_tracking_error_different_returns(self) -> None:
        """Test tracking error with different returns."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.02, 0.03, 0.04] * 100)
        benchmark = pl.Series([0.01, 0.02, 0.03] * 100)

        te = comparator.calculate_tracking_error(strategy, benchmark)

        # Should be positive for different returns
        assert te > 0

    def test_tracking_error_insufficient_data(self) -> None:
        """Test tracking error with insufficient data."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01])
        benchmark = pl.Series([0.01])

        te = comparator.calculate_tracking_error(strategy, benchmark)

        assert te == 0.0

    def test_tracking_error_mismatched_lengths(self) -> None:
        """Test tracking error with mismatched lengths."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.02])

        te = comparator.calculate_tracking_error(strategy, benchmark)

        assert te == 0.0


class TestCalculateInformationRatio:
    """Tests for calculate_information_ratio method."""

    def test_information_ratio_positive(self) -> None:
        """Test information ratio for outperforming strategy."""
        comparator = BenchmarkComparator(risk_free_rate=0.0)
        # Consistently outperform
        strategy = pl.Series([0.02, 0.025, 0.03] * 100)
        benchmark = pl.Series([0.01, 0.015, 0.02] * 100)

        ir = comparator.calculate_information_ratio(strategy, benchmark)

        # Should be positive for outperformance
        assert ir > 0

    def test_information_ratio_negative(self) -> None:
        """Test information ratio for underperforming strategy."""
        comparator = BenchmarkComparator(risk_free_rate=0.0)
        # Consistently underperform with similar volatility
        import random
        random.seed(43)
        benchmark = pl.Series([random.gauss(0.0004, 0.01) for _ in range(252)])
        strategy = pl.Series([r - 0.0002 for r in benchmark])  # Consistently underperform

        ir = comparator.calculate_information_ratio(strategy, benchmark)

        # Should be negative for underperformance
        assert ir < 0

    def test_information_ratio_zero_tracking_error(self) -> None:
        """Test information ratio when tracking error is zero."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.02, 0.03])

        ir = comparator.calculate_information_ratio(strategy, benchmark)

        # Should return 0 when tracking error is zero
        assert ir == 0.0


class TestCalculateCaptureRatios:
    """Tests for calculate_capture_ratios method."""

    def test_capture_ratios_perfect_tracking(self) -> None:
        """Test capture ratios with perfect tracking."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.02, -0.01, 0.03, -0.015])
        benchmark = pl.Series([0.02, -0.01, 0.03, -0.015])

        up_capture, down_capture = comparator.calculate_capture_ratios(strategy, benchmark)

        # Should both be 100% for perfect tracking
        assert abs(up_capture - 100.0) < 1.0
        assert abs(down_capture - 100.0) < 1.0

    def test_capture_ratios_outperformance(self) -> None:
        """Test capture ratios with outperformance."""
        comparator = BenchmarkComparator()
        # Strategy captures more upside, less downside
        strategy = pl.Series([0.03, -0.005, 0.04, -0.008])
        benchmark = pl.Series([0.02, -0.01, 0.03, -0.015])

        up_capture, down_capture = comparator.calculate_capture_ratios(strategy, benchmark)

        # Up capture should be > 100%, down capture < 100%
        assert up_capture > 100.0
        assert down_capture < 100.0

    def test_capture_ratios_underperformance(self) -> None:
        """Test capture ratios with underperformance."""
        comparator = BenchmarkComparator()
        # Strategy captures less upside, more downside
        strategy = pl.Series([0.01, -0.015, 0.015, -0.02])
        benchmark = pl.Series([0.02, -0.01, 0.03, -0.015])

        up_capture, down_capture = comparator.calculate_capture_ratios(strategy, benchmark)

        # Up capture should be < 100%, down capture > 100%
        assert up_capture < 100.0
        assert down_capture > 100.0

    def test_capture_ratios_only_up_markets(self) -> None:
        """Test capture ratios with only up markets."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.02, 0.03])

        up_capture, down_capture = comparator.calculate_capture_ratios(strategy, benchmark)

        # Up capture should be calculated, down capture should be 0
        assert up_capture > 0
        assert down_capture == 0.0

    def test_capture_ratios_only_down_markets(self) -> None:
        """Test capture ratios with only down markets."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([-0.01, -0.02, -0.03])
        benchmark = pl.Series([-0.01, -0.02, -0.03])

        up_capture, down_capture = comparator.calculate_capture_ratios(strategy, benchmark)

        # Down capture should be calculated, up capture should be 0
        assert up_capture == 0.0
        assert down_capture > 0

    def test_capture_ratios_insufficient_data(self) -> None:
        """Test capture ratios with insufficient data."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01])
        benchmark = pl.Series([0.01])

        up_capture, down_capture = comparator.calculate_capture_ratios(strategy, benchmark)

        assert up_capture == 0.0
        assert down_capture == 0.0


class TestCompare:
    """Tests for compare method."""

    def test_compare_valid_benchmark(self) -> None:
        """Test comparing against a valid benchmark."""
        comparator = BenchmarkComparator(risk_free_rate=0.02)
        returns = pl.Series([0.01, 0.02, -0.005] * 100)
        equity = [Decimal("100")] + [Decimal("100") for _ in range(300)]

        comparator.add_benchmark("SPY", returns, equity)

        strategy_returns = pl.Series([0.015, 0.025, -0.003] * 100)
        result = comparator.compare(strategy_returns, "SPY", "TestStrategy")

        assert result.strategy_name == "TestStrategy"
        assert result.benchmark_name == "SPY"
        assert isinstance(result.alpha, float)
        assert isinstance(result.beta, float)
        assert -1.0 <= result.correlation <= 1.0

    def test_compare_nonexistent_benchmark(self) -> None:
        """Test comparing against non-existent benchmark."""
        comparator = BenchmarkComparator()
        strategy_returns = pl.Series([0.01, 0.02, 0.03])

        with pytest.raises(ValueError, match="not found"):
            comparator.compare(strategy_returns, "NONEXISTENT")

    def test_compare_mismatched_lengths(self) -> None:
        """Test comparing with mismatched data lengths."""
        comparator = BenchmarkComparator()
        returns = pl.Series([0.01, 0.02])
        equity = [Decimal("100"), Decimal("101"), Decimal("103")]

        comparator.add_benchmark("SPY", returns, equity)

        strategy_returns = pl.Series([0.01, 0.02, 0.03])  # Different length

        with pytest.raises(ValueError, match="does not match"):
            comparator.compare(strategy_returns, "SPY")

    def test_compare_custom_strategy_name(self) -> None:
        """Test compare with custom strategy name."""
        comparator = BenchmarkComparator()
        returns = pl.Series([0.01, 0.02])
        equity = [Decimal("100"), Decimal("101"), Decimal("103")]

        comparator.add_benchmark("SPY", returns, equity)

        strategy_returns = pl.Series([0.015, 0.025])
        result = comparator.compare(strategy_returns, "SPY", "MyStrategy")

        assert result.strategy_name == "MyStrategy"


class TestCompareAll:
    """Tests for compare_all method."""

    def test_compare_all_multiple_benchmarks(self) -> None:
        """Test comparing against all benchmarks."""
        comparator = BenchmarkComparator(risk_free_rate=0.02)

        # Add multiple benchmarks
        returns1 = pl.Series([0.01, 0.02, -0.005])
        equity1 = [Decimal("100"), Decimal("101"), Decimal("103"), Decimal("102.5")]

        returns2 = pl.Series([0.015, 0.025, -0.003])
        equity2 = [Decimal("100"), Decimal("101.5"), Decimal("104"), Decimal("103.7")]

        comparator.add_benchmark("SPY", returns1, equity1)
        comparator.add_benchmark("QQQ", returns2, equity2)

        strategy_returns = pl.Series([0.012, 0.022, -0.004])
        results = comparator.compare_all(strategy_returns, "TestStrategy")

        assert len(results) == 2
        assert any(r.benchmark_name == "SPY" for r in results)
        assert any(r.benchmark_name == "QQQ" for r in results)

    def test_compare_all_no_benchmarks(self) -> None:
        """Test compare_all with no registered benchmarks."""
        comparator = BenchmarkComparator()
        strategy_returns = pl.Series([0.01, 0.02, 0.03])

        results = comparator.compare_all(strategy_returns)

        assert len(results) == 0

    def test_compare_all_skips_mismatched(self) -> None:
        """Test compare_all skips benchmarks with mismatched lengths."""
        comparator = BenchmarkComparator()

        # Add benchmark with different length
        returns1 = pl.Series([0.01, 0.02])
        equity1 = [Decimal("100"), Decimal("101"), Decimal("103")]

        # Add benchmark with matching length
        returns2 = pl.Series([0.015, 0.025, -0.003])
        equity2 = [Decimal("100"), Decimal("101.5"), Decimal("104"), Decimal("103.7")]

        comparator.add_benchmark("SPY", returns1, equity1)
        comparator.add_benchmark("QQQ", returns2, equity2)

        strategy_returns = pl.Series([0.012, 0.022, -0.004])
        results = comparator.compare_all(strategy_returns)

        # Should only include QQQ (matching length)
        assert len(results) == 1
        assert results[0].benchmark_name == "QQQ"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_zero_returns(self) -> None:
        """Test with all zero returns."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.0, 0.0, 0.0, 0.0])
        benchmark = pl.Series([0.0, 0.0, 0.0, 0.0])

        beta = comparator.calculate_beta(strategy, benchmark)
        corr = comparator.calculate_correlation(strategy, benchmark)
        te = comparator.calculate_tracking_error(strategy, benchmark)

        assert beta == 0.0
        assert corr == 0.0
        assert te < 0.01

    def test_very_large_returns(self) -> None:
        """Test with very large returns."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.5, -0.3, 0.4, -0.2])
        benchmark = pl.Series([0.3, -0.2, 0.25, -0.15])

        beta = comparator.calculate_beta(strategy, benchmark)

        # Should handle large returns without error
        assert isinstance(beta, float)
        assert pytest.approx(beta) != float("inf")

    def test_very_small_returns(self) -> None:
        """Test with very small returns."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.0001, 0.0002, -0.0001])
        benchmark = pl.Series([0.00008, 0.00015, -0.00005])

        beta = comparator.calculate_beta(strategy, benchmark)

        # Should handle small returns without underflow
        assert isinstance(beta, float)

    def test_single_data_point_handled(self) -> None:
        """Test that single data point is handled gracefully."""
        comparator = BenchmarkComparator()
        strategy = pl.Series([0.01])
        benchmark = pl.Series([0.01])

        beta = comparator.calculate_beta(strategy, benchmark)
        alpha = comparator.calculate_alpha(strategy, benchmark)
        corr = comparator.calculate_correlation(strategy, benchmark)

        # All should return 0.0 for insufficient data
        assert beta == 0.0
        assert alpha == 0.0
        assert corr == 0.0
