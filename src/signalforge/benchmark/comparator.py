"""Benchmark comparison and relative performance metrics.

This module provides tools for comparing trading strategy performance against
standard market benchmarks (SPY, QQQ, IWM) and buy-and-hold strategies.
It calculates key relative performance metrics including alpha, beta,
correlation, tracking error, information ratio, and capture ratios.

All calculations use standard financial formulas and are optimized for
performance with Polars Series operations.

Examples:
    Compare strategy against S&P 500:

    >>> import polars as pl
    >>> from decimal import Decimal
    >>> from signalforge.benchmark.comparator import BenchmarkComparator
    >>>
    >>> comparator = BenchmarkComparator(risk_free_rate=0.02)
    >>> comparator.add_benchmark(
    ...     name="SPY",
    ...     returns=pl.Series([0.01, -0.005, 0.015]),
    ...     equity_curve=[Decimal("100000"), Decimal("101000"), Decimal("100500")]
    ... )
    >>> strategy_returns = pl.Series([0.02, -0.003, 0.025])
    >>> result = comparator.compare(strategy_returns, "SPY")
    >>> print(f"Alpha: {result.alpha:.4f}, Beta: {result.beta:.4f}")

    Compare against all benchmarks:

    >>> results = comparator.compare_all(strategy_returns)
    >>> for r in results:
    ...     print(f"{r.benchmark_name}: IR={r.information_ratio:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class BenchmarkData:
    """Data for a benchmark index.

    This dataclass stores the name, returns series, and equity curve
    for a benchmark index used in relative performance comparison.

    Attributes:
        name: Identifier for the benchmark (e.g., "SPY", "QQQ")
        returns: Series of period returns (as decimals, e.g., 0.01 for 1%)
        equity_curve: List of equity values over time as Decimal for precision

    Note:
        The length of returns should be len(equity_curve) - 1, as returns
        are calculated from consecutive equity curve values.
    """

    name: str
    returns: pl.Series
    equity_curve: list[Decimal]

    def __post_init__(self) -> None:
        """Validate benchmark data after initialization."""
        if not self.name:
            raise ValueError("Benchmark name cannot be empty")
        if self.returns.is_empty():
            raise ValueError(f"Benchmark {self.name} has empty returns")
        if len(self.equity_curve) < 2:
            raise ValueError(f"Benchmark {self.name} equity curve must have at least 2 points")
        if len(self.returns) != len(self.equity_curve) - 1:
            raise ValueError(
                f"Benchmark {self.name}: returns length ({len(self.returns)}) "
                f"must equal equity_curve length - 1 ({len(self.equity_curve) - 1})"
            )


@dataclass
class ComparisonResult:
    """Results from comparing a strategy against a benchmark.

    This dataclass encapsulates all relative performance metrics calculated
    when comparing a trading strategy against a benchmark index.

    Attributes:
        strategy_name: Name of the strategy being compared
        benchmark_name: Name of the benchmark used for comparison
        alpha: Annualized excess return over the benchmark (as percentage)
        beta: Systematic risk relative to the benchmark (unitless)
        correlation: Correlation coefficient between strategy and benchmark returns [-1, 1]
        tracking_error: Annualized standard deviation of excess returns (as percentage)
        information_ratio: Risk-adjusted excess return (alpha / tracking_error)
        up_capture: Percentage of benchmark's gains captured during up periods
        down_capture: Percentage of benchmark's losses captured during down periods
        relative_drawdown: Maximum drawdown relative to the benchmark (as percentage)

    Note:
        - Alpha > 0 indicates outperformance
        - Beta > 1 indicates higher volatility than benchmark
        - Correlation near 1 indicates high similarity in movement
        - IR > 0.5 is considered good, > 1.0 is excellent
        - Up capture > 100% is desirable, down capture < 100% is desirable
    """

    strategy_name: str
    benchmark_name: str
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    relative_drawdown: float

    def __post_init__(self) -> None:
        """Validate comparison results after initialization."""
        if not self.strategy_name:
            raise ValueError("strategy_name cannot be empty")
        if not self.benchmark_name:
            raise ValueError("benchmark_name cannot be empty")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError(f"correlation must be in [-1, 1], got {self.correlation}")
        if self.tracking_error < 0:
            raise ValueError(f"tracking_error cannot be negative, got {self.tracking_error}")

    def to_dict(self) -> dict[str, str | float]:
        """Convert comparison results to dictionary.

        Returns:
            Dictionary with metric names as keys and values.
        """
        return {
            "strategy_name": self.strategy_name,
            "benchmark_name": self.benchmark_name,
            "alpha": self.alpha,
            "beta": self.beta,
            "correlation": self.correlation,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "up_capture": self.up_capture,
            "down_capture": self.down_capture,
            "relative_drawdown": self.relative_drawdown,
        }


class BenchmarkComparator:
    """Compare strategy performance against market benchmarks.

    This class provides comprehensive benchmark comparison capabilities,
    including predefined benchmarks (SPY, QQQ, IWM) and custom benchmarks.
    It calculates alpha, beta, correlation, tracking error, information ratio,
    and capture ratios using standard financial formulas.

    Attributes:
        risk_free_rate: Annual risk-free rate used for alpha calculation (as decimal)
        benchmarks: Dictionary of registered benchmarks

    Examples:
        Basic usage with predefined benchmark:

        >>> import polars as pl
        >>> from signalforge.benchmark.comparator import BenchmarkComparator
        >>>
        >>> comparator = BenchmarkComparator(risk_free_rate=0.02)
        >>> comparator.add_benchmark(
        ...     name="SPY",
        ...     returns=pl.Series([0.01, -0.005, 0.015]),
        ...     equity_curve=[Decimal("100"), Decimal("101"), Decimal("100.5")]
        ... )
        >>> strategy_returns = pl.Series([0.02, -0.003, 0.025])
        >>> result = comparator.compare(strategy_returns, "SPY")
        >>> print(f"Alpha: {result.alpha:.4f}")

        Compare against multiple benchmarks:

        >>> results = comparator.compare_all(strategy_returns)
        >>> best = max(results, key=lambda r: r.information_ratio)
        >>> print(f"Best IR vs {best.benchmark_name}: {best.information_ratio:.2f}")
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """Initialize benchmark comparator.

        Args:
            risk_free_rate: Annual risk-free rate (as decimal, e.g., 0.02 for 2%).
                          Used in alpha and information ratio calculations.
                          Defaults to 0.0.

        Raises:
            ValueError: If risk_free_rate is negative or unreasonably high (>1.0)
        """
        if risk_free_rate < 0:
            raise ValueError(f"risk_free_rate cannot be negative, got {risk_free_rate}")
        if risk_free_rate > 1.0:
            raise ValueError(f"risk_free_rate seems unreasonably high, got {risk_free_rate}")

        self.risk_free_rate = risk_free_rate
        self.benchmarks: dict[str, BenchmarkData] = {}
        logger.info("BenchmarkComparator initialized", risk_free_rate=risk_free_rate)

    def add_benchmark(
        self, name: str, returns: pl.Series, equity_curve: list[Decimal]
    ) -> None:
        """Register a benchmark for comparison.

        Args:
            name: Unique identifier for the benchmark (e.g., "SPY", "QQQ")
            returns: Series of period returns (as decimals)
            equity_curve: List of equity values over time

        Raises:
            ValueError: If benchmark with same name already exists or data is invalid

        Examples:
            >>> comparator = BenchmarkComparator()
            >>> comparator.add_benchmark(
            ...     name="SPY",
            ...     returns=pl.Series([0.01, 0.02, -0.005]),
            ...     equity_curve=[Decimal("100"), Decimal("101"), Decimal("103.02")]
            ... )
        """
        if name in self.benchmarks:
            raise ValueError(f"Benchmark {name} already exists")

        benchmark = BenchmarkData(name=name, returns=returns, equity_curve=equity_curve)
        self.benchmarks[name] = benchmark
        logger.info("Benchmark added", name=name, data_points=len(returns))

    def calculate_alpha(
        self, strategy_returns: pl.Series, benchmark_returns: pl.Series
    ) -> float:
        """Calculate Jensen's alpha (annualized excess return).

        Alpha measures the excess return of the strategy over the return
        predicted by the Capital Asset Pricing Model (CAPM). It represents
        the value added by active management.

        Formula:
            Alpha = Strategy_Return - (Rf + Beta * (Benchmark_Return - Rf))

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            benchmark_returns: Series of benchmark period returns (as decimals)

        Returns:
            Annualized alpha as percentage. Positive values indicate outperformance.

        Note:
            Returns 0.0 if series are mismatched or have insufficient data.

        Examples:
            >>> import polars as pl
            >>> comparator = BenchmarkComparator(risk_free_rate=0.02)
            >>> strategy = pl.Series([0.02, 0.01, 0.03])
            >>> benchmark = pl.Series([0.01, 0.005, 0.015])
            >>> alpha = comparator.calculate_alpha(strategy, benchmark)
            >>> print(f"Alpha: {alpha:.2f}%")
        """
        if strategy_returns.len() != benchmark_returns.len():
            logger.warning(
                "calculate_alpha: returns series length mismatch",
                strategy_len=strategy_returns.len(),
                benchmark_len=benchmark_returns.len(),
            )
            return 0.0

        if strategy_returns.is_empty() or strategy_returns.len() < 2:
            logger.warning("calculate_alpha: insufficient data")
            return 0.0

        # Calculate beta first
        beta = self.calculate_beta(strategy_returns, benchmark_returns)

        # Calculate annualized returns (assuming daily returns)
        strategy_mean = strategy_returns.drop_nulls().mean()
        benchmark_mean = benchmark_returns.drop_nulls().mean()

        if strategy_mean is None or benchmark_mean is None:
            return 0.0

        strategy_annual = float(strategy_mean) * 252.0 * 100.0  # type: ignore[arg-type]
        benchmark_annual = float(benchmark_mean) * 252.0 * 100.0  # type: ignore[arg-type]

        # CAPM: Alpha = Rs - (Rf + Beta * (Rb - Rf))
        rf = self.risk_free_rate * 100.0  # Convert to percentage
        expected_return = rf + beta * (benchmark_annual - rf)
        alpha = strategy_annual - expected_return

        return float(alpha)

    def calculate_beta(
        self, strategy_returns: pl.Series, benchmark_returns: pl.Series
    ) -> float:
        """Calculate beta (systematic risk).

        Beta measures the sensitivity of the strategy's returns to the
        benchmark's returns. It represents systematic risk that cannot
        be diversified away.

        Formula:
            Beta = Cov(Strategy, Benchmark) / Var(Benchmark)

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            benchmark_returns: Series of benchmark period returns (as decimals)

        Returns:
            Beta coefficient. Values > 1 indicate higher volatility than benchmark,
            < 1 indicate lower volatility. Returns 0.0 if insufficient data.

        Note:
            Beta = 1.0 means same volatility as benchmark
            Beta > 1.0 means more volatile (amplified movements)
            Beta < 1.0 means less volatile (dampened movements)

        Examples:
            >>> import polars as pl
            >>> comparator = BenchmarkComparator()
            >>> strategy = pl.Series([0.02, -0.01, 0.03])
            >>> benchmark = pl.Series([0.01, -0.005, 0.015])
            >>> beta = comparator.calculate_beta(strategy, benchmark)
            >>> print(f"Beta: {beta:.2f}")
        """
        if strategy_returns.len() != benchmark_returns.len():
            logger.warning(
                "calculate_beta: returns series length mismatch",
                strategy_len=strategy_returns.len(),
                benchmark_len=benchmark_returns.len(),
            )
            return 0.0

        if strategy_returns.is_empty() or strategy_returns.len() < 2:
            logger.warning("calculate_beta: insufficient data")
            return 0.0

        # Create DataFrame to handle nulls consistently across both series
        df = pl.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns})
        df_clean = df.drop_nulls()

        if df_clean.height < 2:
            return 0.0

        strategy_clean = df_clean["strategy"]
        benchmark_clean = df_clean["benchmark"]

        # Calculate means
        strategy_mean = strategy_clean.mean()
        benchmark_mean = benchmark_clean.mean()

        if strategy_mean is None or benchmark_mean is None:
            return 0.0

        strategy_centered = strategy_clean - float(strategy_mean)  # type: ignore[arg-type]
        benchmark_centered = benchmark_clean - float(benchmark_mean)  # type: ignore[arg-type]

        # Calculate covariance and variance using sample formulas (ddof=1)
        n = df_clean.height
        covariance = (strategy_centered * benchmark_centered).sum() / (n - 1)
        benchmark_var = (benchmark_centered * benchmark_centered).sum() / (n - 1)

        if benchmark_var is None or float(benchmark_var) == 0.0:
            return 0.0

        if covariance is None:
            return 0.0

        beta = float(covariance) / float(benchmark_var)

        return float(beta)

    def calculate_correlation(
        self, strategy_returns: pl.Series, benchmark_returns: pl.Series
    ) -> float:
        """Calculate Pearson correlation coefficient.

        Correlation measures the linear relationship between strategy and
        benchmark returns. It indicates how closely the strategy tracks
        the benchmark's movements.

        Formula:
            Correlation = Cov(Strategy, Benchmark) / (Std(Strategy) * Std(Benchmark))

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            benchmark_returns: Series of benchmark period returns (as decimals)

        Returns:
            Correlation coefficient in range [-1, 1].
            1.0 = perfect positive correlation
            0.0 = no correlation
            -1.0 = perfect negative correlation
            Returns 0.0 if insufficient data.

        Examples:
            >>> import polars as pl
            >>> comparator = BenchmarkComparator()
            >>> strategy = pl.Series([0.02, -0.01, 0.03])
            >>> benchmark = pl.Series([0.02, -0.01, 0.03])
            >>> corr = comparator.calculate_correlation(strategy, benchmark)
            >>> print(f"Correlation: {corr:.2f}")
        """
        if strategy_returns.len() != benchmark_returns.len():
            logger.warning(
                "calculate_correlation: returns series length mismatch",
                strategy_len=strategy_returns.len(),
                benchmark_len=benchmark_returns.len(),
            )
            return 0.0

        if strategy_returns.is_empty() or strategy_returns.len() < 2:
            logger.warning("calculate_correlation: insufficient data")
            return 0.0

        # Create DataFrame to handle nulls consistently across both series
        df = pl.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns})
        df_clean = df.drop_nulls()

        if df_clean.height < 2:
            return 0.0

        strategy_clean = df_clean["strategy"]
        benchmark_clean = df_clean["benchmark"]

        # Calculate means
        strategy_mean = strategy_clean.mean()
        benchmark_mean = benchmark_clean.mean()

        if strategy_mean is None or benchmark_mean is None:
            return 0.0

        strategy_centered = strategy_clean - float(strategy_mean)  # type: ignore[arg-type]
        benchmark_centered = benchmark_clean - float(benchmark_mean)  # type: ignore[arg-type]

        # Calculate standard deviations using sample formula (ddof=1)
        n = df_clean.height
        strategy_std = ((strategy_centered * strategy_centered).sum() / (n - 1)) ** 0.5
        benchmark_std = ((benchmark_centered * benchmark_centered).sum() / (n - 1)) ** 0.5

        if (
            strategy_std is None
            or benchmark_std is None
            or float(strategy_std) == 0.0
            or float(benchmark_std) == 0.0
        ):
            return 0.0

        # Calculate covariance using sample formula (ddof=1)
        covariance = (strategy_centered * benchmark_centered).sum() / (n - 1)
        if covariance is None:
            return 0.0

        correlation = float(covariance) / (float(strategy_std) * float(benchmark_std))

        # Clamp to [-1, 1] to handle floating point errors
        correlation = max(-1.0, min(1.0, correlation))

        return float(correlation)

    def calculate_tracking_error(
        self, strategy_returns: pl.Series, benchmark_returns: pl.Series
    ) -> float:
        """Calculate tracking error (annualized).

        Tracking error measures the consistency of excess returns by
        calculating the standard deviation of the difference between
        strategy and benchmark returns.

        Formula:
            Tracking Error = Std(Strategy Returns - Benchmark Returns) * sqrt(252)

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            benchmark_returns: Series of benchmark period returns (as decimals)

        Returns:
            Annualized tracking error as percentage. Higher values indicate
            less consistent tracking. Returns 0.0 if insufficient data.

        Note:
            Low tracking error (<2%) suggests passive/index strategy
            Medium tracking error (2-5%) suggests enhanced index strategy
            High tracking error (>5%) suggests active management

        Examples:
            >>> import polars as pl
            >>> comparator = BenchmarkComparator()
            >>> strategy = pl.Series([0.02, -0.01, 0.03])
            >>> benchmark = pl.Series([0.015, -0.008, 0.025])
            >>> te = comparator.calculate_tracking_error(strategy, benchmark)
            >>> print(f"Tracking Error: {te:.2f}%")
        """
        if strategy_returns.len() != benchmark_returns.len():
            logger.warning(
                "calculate_tracking_error: returns series length mismatch",
                strategy_len=strategy_returns.len(),
                benchmark_len=benchmark_returns.len(),
            )
            return 0.0

        if strategy_returns.is_empty() or strategy_returns.len() < 2:
            logger.warning("calculate_tracking_error: insufficient data")
            return 0.0

        # Calculate excess returns
        excess_returns = strategy_returns - benchmark_returns

        # Drop nulls
        excess_clean = excess_returns.drop_nulls()

        if excess_clean.len() < 2:
            return 0.0

        # Calculate standard deviation of excess returns
        excess_std = excess_clean.std()
        if excess_std is None:
            return 0.0

        # Annualize (assuming daily returns)
        tracking_error = float(excess_std) * (252.0**0.5) * 100.0  # type: ignore[arg-type]

        return float(tracking_error)

    def calculate_information_ratio(
        self, strategy_returns: pl.Series, benchmark_returns: pl.Series
    ) -> float:
        """Calculate information ratio.

        Information ratio measures risk-adjusted excess return by dividing
        alpha (excess return) by tracking error (excess return volatility).
        It's similar to Sharpe ratio but measures performance relative to
        a benchmark rather than the risk-free rate.

        Formula:
            Information Ratio = Alpha / Tracking Error

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            benchmark_returns: Series of benchmark period returns (as decimals)

        Returns:
            Information ratio. Higher values indicate better risk-adjusted
            outperformance. Returns 0.0 if tracking error is zero or insufficient data.

        Note:
            IR > 0.5 is considered good
            IR > 1.0 is considered excellent
            IR < 0 indicates underperformance

        Examples:
            >>> import polars as pl
            >>> comparator = BenchmarkComparator()
            >>> strategy = pl.Series([0.02, -0.01, 0.03])
            >>> benchmark = pl.Series([0.015, -0.008, 0.025])
            >>> ir = comparator.calculate_information_ratio(strategy, benchmark)
            >>> print(f"Information Ratio: {ir:.2f}")
        """
        tracking_error = self.calculate_tracking_error(strategy_returns, benchmark_returns)

        if tracking_error == 0.0:
            logger.warning("calculate_information_ratio: tracking error is zero")
            return 0.0

        alpha = self.calculate_alpha(strategy_returns, benchmark_returns)

        information_ratio = alpha / tracking_error

        return float(information_ratio)

    def calculate_capture_ratios(
        self, strategy_returns: pl.Series, benchmark_returns: pl.Series
    ) -> tuple[float, float]:
        """Calculate up-capture and down-capture ratios.

        Capture ratios measure how much of the benchmark's movement is captured
        by the strategy during up and down markets separately.

        Up-capture: (Strategy avg return in up markets) / (Benchmark avg return in up markets)
        Down-capture: (Strategy avg return in down markets) / (Benchmark avg return in down markets)

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            benchmark_returns: Series of benchmark period returns (as decimals)

        Returns:
            Tuple of (up_capture, down_capture) as percentages.
            - up_capture > 100% means strategy captures more upside than benchmark
            - down_capture < 100% means strategy captures less downside than benchmark
            Returns (0.0, 0.0) if insufficient data.

        Note:
            Ideal strategy has up_capture > 100% and down_capture < 100%

        Examples:
            >>> import polars as pl
            >>> comparator = BenchmarkComparator()
            >>> strategy = pl.Series([0.02, -0.005, 0.03])
            >>> benchmark = pl.Series([0.015, -0.01, 0.025])
            >>> up, down = comparator.calculate_capture_ratios(strategy, benchmark)
            >>> print(f"Up: {up:.1f}%, Down: {down:.1f}%")
        """
        if strategy_returns.len() != benchmark_returns.len():
            logger.warning(
                "calculate_capture_ratios: returns series length mismatch",
                strategy_len=strategy_returns.len(),
                benchmark_len=benchmark_returns.len(),
            )
            return 0.0, 0.0

        if strategy_returns.is_empty() or strategy_returns.len() < 2:
            logger.warning("calculate_capture_ratios: insufficient data")
            return 0.0, 0.0

        # Create DataFrame to handle nulls consistently across both series
        df = pl.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns})
        df_clean = df.drop_nulls()

        if df_clean.height < 2:
            return 0.0, 0.0

        strategy_clean = df_clean["strategy"]
        benchmark_clean = df_clean["benchmark"]

        # Filter for up markets (benchmark positive)
        up_mask = benchmark_clean > 0
        strategy_up = strategy_clean.filter(up_mask)
        benchmark_up = benchmark_clean.filter(up_mask)

        # Filter for down markets (benchmark negative)
        down_mask = benchmark_clean < 0
        strategy_down = strategy_clean.filter(down_mask)
        benchmark_down = benchmark_clean.filter(down_mask)

        # Calculate up-capture
        up_capture = 0.0
        if benchmark_up.len() > 0:
            strategy_up_mean = strategy_up.mean()
            benchmark_up_mean = benchmark_up.mean()
            if (
                strategy_up_mean is not None
                and benchmark_up_mean is not None
                and float(benchmark_up_mean) != 0.0  # type: ignore[arg-type]
            ):
                up_capture = (
                    float(strategy_up_mean)  # type: ignore[arg-type]
                    / float(benchmark_up_mean)  # type: ignore[arg-type]
                    * 100.0
                )

        # Calculate down-capture
        down_capture = 0.0
        if benchmark_down.len() > 0:
            strategy_down_mean = strategy_down.mean()
            benchmark_down_mean = benchmark_down.mean()
            if (
                strategy_down_mean is not None
                and benchmark_down_mean is not None
                and float(benchmark_down_mean) != 0.0  # type: ignore[arg-type]
            ):
                down_capture = (
                    float(strategy_down_mean)  # type: ignore[arg-type]
                    / float(benchmark_down_mean)  # type: ignore[arg-type]
                    * 100.0
                )

        return float(up_capture), float(down_capture)

    def compare(
        self, strategy_returns: pl.Series, benchmark_name: str, strategy_name: str = "Strategy"
    ) -> ComparisonResult:
        """Compare strategy against a specific benchmark.

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            benchmark_name: Name of registered benchmark to compare against
            strategy_name: Name for the strategy (default: "Strategy")

        Returns:
            ComparisonResult containing all relative performance metrics

        Raises:
            ValueError: If benchmark_name not found or data length mismatch

        Examples:
            >>> import polars as pl
            >>> from decimal import Decimal
            >>> comparator = BenchmarkComparator(risk_free_rate=0.02)
            >>> comparator.add_benchmark(
            ...     "SPY",
            ...     pl.Series([0.01, 0.02]),
            ...     [Decimal("100"), Decimal("101"), Decimal("103.02")]
            ... )
            >>> result = comparator.compare(pl.Series([0.015, 0.025]), "SPY")
            >>> print(f"Alpha: {result.alpha:.2f}%")
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark {benchmark_name} not found")

        benchmark = self.benchmarks[benchmark_name]
        benchmark_returns = benchmark.returns

        if strategy_returns.len() != benchmark_returns.len():
            raise ValueError(
                f"Strategy returns length ({strategy_returns.len()}) "
                f"does not match benchmark returns length ({benchmark_returns.len()})"
            )

        logger.info(
            "Comparing strategy to benchmark",
            strategy_name=strategy_name,
            benchmark_name=benchmark_name,
        )

        # Calculate all metrics
        alpha = self.calculate_alpha(strategy_returns, benchmark_returns)
        beta = self.calculate_beta(strategy_returns, benchmark_returns)
        correlation = self.calculate_correlation(strategy_returns, benchmark_returns)
        tracking_error = self.calculate_tracking_error(strategy_returns, benchmark_returns)
        information_ratio = self.calculate_information_ratio(strategy_returns, benchmark_returns)
        up_capture, down_capture = self.calculate_capture_ratios(
            strategy_returns, benchmark_returns
        )

        # Calculate relative drawdown (simplified as 0.0 for now)
        # This would require equity curves for proper calculation
        relative_drawdown = 0.0

        result = ComparisonResult(
            strategy_name=strategy_name,
            benchmark_name=benchmark_name,
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            up_capture=up_capture,
            down_capture=down_capture,
            relative_drawdown=relative_drawdown,
        )

        logger.info(
            "Comparison complete",
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            information_ratio=information_ratio,
        )

        return result

    def compare_all(
        self, strategy_returns: pl.Series, strategy_name: str = "Strategy"
    ) -> list[ComparisonResult]:
        """Compare strategy against all registered benchmarks.

        Args:
            strategy_returns: Series of strategy period returns (as decimals)
            strategy_name: Name for the strategy (default: "Strategy")

        Returns:
            List of ComparisonResult objects, one for each benchmark

        Note:
            Only benchmarks with matching data length will be compared.
            Mismatched benchmarks will be logged and skipped.

        Examples:
            >>> import polars as pl
            >>> comparator = BenchmarkComparator()
            >>> # ... add multiple benchmarks ...
            >>> results = comparator.compare_all(pl.Series([0.01, 0.02, 0.015]))
            >>> best = max(results, key=lambda r: r.information_ratio)
            >>> print(f"Best vs {best.benchmark_name}: IR={best.information_ratio:.2f}")
        """
        if not self.benchmarks:
            logger.warning("compare_all: no benchmarks registered")
            return []

        results = []
        for benchmark_name in self.benchmarks:
            try:
                result = self.compare(strategy_returns, benchmark_name, strategy_name)
                results.append(result)
            except ValueError as e:
                logger.warning(
                    "Skipping benchmark due to error",
                    benchmark_name=benchmark_name,
                    error=str(e),
                )
                continue

        logger.info("Compared against all benchmarks", num_results=len(results))
        return results
