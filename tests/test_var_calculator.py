"""Comprehensive tests for VaR calculator implementation.

Test coverage includes:
- Configuration validation
- Return calculations
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Portfolio VaR with correlations
- Expected Shortfall (CVaR)
- Edge cases: empty data, constant returns, single values, etc.
- Validation error handling
"""

from decimal import Decimal

import polars as pl
import pytest

from signalforge.core.exceptions import ValidationError
from signalforge.risk.var_calculator import (
    PositionVaR,
    VaRCalculator,
    VaRConfig,
    VaRMethod,
    VaRResult,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> VaRConfig:
    """Default VaR configuration."""
    return VaRConfig(confidence_level=0.95, horizon_days=1, method=VaRMethod.HISTORICAL)


@pytest.fixture
def sample_prices() -> pl.Series:
    """Sample price series with realistic market data."""
    prices = [
        100.0,
        101.5,
        99.8,
        102.3,
        101.0,
        103.5,
        102.8,
        104.2,
        103.0,
        105.5,
        104.3,
        106.0,
        105.2,
        107.1,
        106.5,
        108.3,
        107.8,
        109.5,
        108.9,
        110.2,
        109.5,
        111.3,
        110.8,
        112.5,
        111.9,
        113.6,
        112.8,
        114.5,
        113.7,
        115.2,
    ]
    return pl.Series("prices", prices, dtype=pl.Float64)


@pytest.fixture
def sample_returns(sample_prices: pl.Series) -> pl.Series:
    """Sample returns derived from prices."""
    calculator = VaRCalculator(VaRConfig())
    return calculator.calculate_returns(sample_prices)


@pytest.fixture
def volatile_prices() -> pl.Series:
    """Highly volatile price series."""
    prices = [
        100.0,
        110.0,
        95.0,
        105.0,
        90.0,
        115.0,
        85.0,
        120.0,
        80.0,
        125.0,
        75.0,
        130.0,
        70.0,
        135.0,
        65.0,
        140.0,
        60.0,
        145.0,
        55.0,
        150.0,
    ]
    return pl.Series("prices", prices, dtype=pl.Float64)


@pytest.fixture
def constant_prices() -> pl.Series:
    """Constant price series (zero volatility edge case)."""
    return pl.Series("prices", [100.0] * 50, dtype=pl.Float64)


@pytest.fixture
def portfolio_value() -> Decimal:
    """Standard portfolio value for testing."""
    return Decimal("1000000")


# ============================================================================
# VaRConfig Tests
# ============================================================================


class TestVaRConfig:
    """Tests for VaRConfig dataclass validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VaRConfig()
        assert config.confidence_level == 0.95
        assert config.horizon_days == 1
        assert config.method == VaRMethod.HISTORICAL

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = VaRConfig(
            confidence_level=0.99, horizon_days=10, method=VaRMethod.PARAMETRIC
        )
        assert config.confidence_level == 0.99
        assert config.horizon_days == 10
        assert config.method == VaRMethod.PARAMETRIC

    def test_invalid_confidence_level_too_low(self) -> None:
        """Test validation of confidence level that is too low."""
        with pytest.raises(ValidationError) as exc_info:
            VaRConfig(confidence_level=0.0)
        assert exc_info.value.details.get("field") == "confidence_level"

    def test_invalid_confidence_level_too_high(self) -> None:
        """Test validation of confidence level that is too high."""
        with pytest.raises(ValidationError) as exc_info:
            VaRConfig(confidence_level=1.0)
        assert exc_info.value.details.get("field") == "confidence_level"

    def test_invalid_confidence_level_negative(self) -> None:
        """Test validation of negative confidence level."""
        with pytest.raises(ValidationError) as exc_info:
            VaRConfig(confidence_level=-0.5)
        assert exc_info.value.details.get("field") == "confidence_level"

    def test_invalid_confidence_level_above_one(self) -> None:
        """Test validation of confidence level above 1."""
        with pytest.raises(ValidationError) as exc_info:
            VaRConfig(confidence_level=1.5)
        assert exc_info.value.details.get("field") == "confidence_level"

    def test_invalid_horizon_days_zero(self) -> None:
        """Test validation of zero horizon days."""
        with pytest.raises(ValidationError) as exc_info:
            VaRConfig(horizon_days=0)
        assert exc_info.value.details.get("field") == "horizon_days"

    def test_invalid_horizon_days_negative(self) -> None:
        """Test validation of negative horizon days."""
        with pytest.raises(ValidationError) as exc_info:
            VaRConfig(horizon_days=-5)
        assert exc_info.value.details.get("field") == "horizon_days"

    def test_valid_edge_confidence_levels(self) -> None:
        """Test valid edge case confidence levels."""
        config1 = VaRConfig(confidence_level=0.01)
        assert config1.confidence_level == 0.01

        config2 = VaRConfig(confidence_level=0.99)
        assert config2.confidence_level == 0.99

    def test_config_immutability(self) -> None:
        """Test that config is immutable (frozen)."""
        config = VaRConfig()
        with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError
            config.confidence_level = 0.99  # type: ignore[misc]


# ============================================================================
# VaRResult Tests
# ============================================================================


class TestVaRResult:
    """Tests for VaRResult dataclass."""

    def test_var_result_creation(self) -> None:
        """Test creating a VaR result."""
        result = VaRResult(
            var_amount=Decimal("50000"),
            var_pct=0.05,
            confidence_level=0.95,
            horizon_days=1,
            method=VaRMethod.HISTORICAL,
            expected_shortfall=Decimal("65000"),
        )
        assert result.var_amount == Decimal("50000")
        assert result.var_pct == 0.05
        assert result.confidence_level == 0.95
        assert result.horizon_days == 1
        assert result.method == VaRMethod.HISTORICAL
        assert result.expected_shortfall == Decimal("65000")

    def test_var_result_immutability(self) -> None:
        """Test that VaR result is immutable."""
        result = VaRResult(
            var_amount=Decimal("50000"),
            var_pct=0.05,
            confidence_level=0.95,
            horizon_days=1,
            method=VaRMethod.HISTORICAL,
            expected_shortfall=Decimal("65000"),
        )
        with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError
            result.var_amount = Decimal("60000")  # type: ignore[misc]


# ============================================================================
# PositionVaR Tests
# ============================================================================


class TestPositionVaR:
    """Tests for PositionVaR dataclass."""

    def test_position_var_creation(self) -> None:
        """Test creating a position VaR."""
        position = PositionVaR(symbol="AAPL", weight=0.5, var_individual=Decimal("25000"))
        assert position.symbol == "AAPL"
        assert position.weight == 0.5
        assert position.var_individual == Decimal("25000")

    def test_position_var_valid_edge_weights(self) -> None:
        """Test valid edge case weights."""
        position1 = PositionVaR(symbol="AAPL", weight=0.0, var_individual=Decimal("0"))
        assert position1.weight == 0.0

        position2 = PositionVaR(symbol="AAPL", weight=1.0, var_individual=Decimal("100000"))
        assert position2.weight == 1.0

    def test_position_var_invalid_weight_negative(self) -> None:
        """Test validation of negative weight."""
        with pytest.raises(ValidationError) as exc_info:
            PositionVaR(symbol="AAPL", weight=-0.1, var_individual=Decimal("25000"))
        assert "weight" in str(exc_info.value)

    def test_position_var_invalid_weight_above_one(self) -> None:
        """Test validation of weight above 1."""
        with pytest.raises(ValidationError) as exc_info:
            PositionVaR(symbol="AAPL", weight=1.5, var_individual=Decimal("25000"))
        assert "weight" in str(exc_info.value)


# ============================================================================
# Calculate Returns Tests
# ============================================================================


class TestCalculateReturns:
    """Tests for return calculation."""

    def test_calculate_returns_normal_prices(
        self, default_config: VaRConfig, sample_prices: pl.Series
    ) -> None:
        """Test return calculation with normal price series."""
        calculator = VaRCalculator(default_config)
        returns = calculator.calculate_returns(sample_prices)

        # Should have one less return than prices
        assert len(returns) == len(sample_prices) - 1

        # Returns should be log returns
        # Verify first return: ln(101.5/100.0) ≈ 0.0149
        first_return = float(returns[0])
        expected = pl.Series([101.5 / 100.0]).log()[0]
        assert abs(first_return - float(expected)) < 1e-6

    def test_calculate_returns_empty_series(self, default_config: VaRConfig) -> None:
        """Test return calculation with empty price series."""
        calculator = VaRCalculator(default_config)
        empty_prices = pl.Series("prices", [], dtype=pl.Float64)
        returns = calculator.calculate_returns(empty_prices)

        assert len(returns) == 0

    def test_calculate_returns_single_price(self, default_config: VaRConfig) -> None:
        """Test return calculation with single price."""
        calculator = VaRCalculator(default_config)
        single_price = pl.Series("prices", [100.0], dtype=pl.Float64)
        returns = calculator.calculate_returns(single_price)

        # Single price produces zero returns
        assert len(returns) == 0

    def test_calculate_returns_two_prices(self, default_config: VaRConfig) -> None:
        """Test return calculation with two prices."""
        calculator = VaRCalculator(default_config)
        two_prices = pl.Series("prices", [100.0, 105.0], dtype=pl.Float64)
        returns = calculator.calculate_returns(two_prices)

        assert len(returns) == 1
        import math

        expected_return = math.log(105.0 / 100.0)
        assert abs(float(returns[0]) - expected_return) < 1e-10

    def test_calculate_returns_constant_prices(
        self, default_config: VaRConfig, constant_prices: pl.Series
    ) -> None:
        """Test return calculation with constant prices (zero returns)."""
        calculator = VaRCalculator(default_config)
        returns = calculator.calculate_returns(constant_prices)

        assert len(returns) == len(constant_prices) - 1
        # All returns should be zero (log(1) = 0)
        assert all(abs(float(r)) < 1e-10 for r in returns)

    def test_calculate_returns_negative_prices(self, default_config: VaRConfig) -> None:
        """Test return calculation with negative prices raises error."""
        calculator = VaRCalculator(default_config)
        invalid_prices = pl.Series("prices", [100.0, -50.0, 75.0], dtype=pl.Float64)

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_returns(invalid_prices)
        assert "positive" in str(exc_info.value).lower()

    def test_calculate_returns_zero_prices(self, default_config: VaRConfig) -> None:
        """Test return calculation with zero prices raises error."""
        calculator = VaRCalculator(default_config)
        invalid_prices = pl.Series("prices", [100.0, 0.0, 75.0], dtype=pl.Float64)

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_returns(invalid_prices)
        assert "positive" in str(exc_info.value).lower()


# ============================================================================
# Historical VaR Tests
# ============================================================================


class TestHistoricalVaR:
    """Tests for historical VaR calculation."""

    def test_historical_var_normal_returns(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test historical VaR with normal returns."""
        config = VaRConfig(confidence_level=0.95, method=VaRMethod.HISTORICAL)
        calculator = VaRCalculator(config)

        result = calculator.calculate_historical_var(sample_returns, portfolio_value)

        assert isinstance(result, VaRResult)
        assert result.var_amount > 0
        assert result.var_pct > 0
        assert result.confidence_level == 0.95
        assert result.horizon_days == 1
        assert result.method == VaRMethod.HISTORICAL
        assert result.expected_shortfall >= result.var_amount

    def test_historical_var_different_confidence_levels(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test that VaR increases with confidence level."""
        config_95 = VaRConfig(confidence_level=0.95, method=VaRMethod.HISTORICAL)
        config_99 = VaRConfig(confidence_level=0.99, method=VaRMethod.HISTORICAL)

        calculator_95 = VaRCalculator(config_95)
        calculator_99 = VaRCalculator(config_99)

        result_95 = calculator_95.calculate_historical_var(sample_returns, portfolio_value)
        result_99 = calculator_99.calculate_historical_var(sample_returns, portfolio_value)

        # 99% VaR should be higher than 95% VaR
        assert result_99.var_amount >= result_95.var_amount

    def test_historical_var_multi_day_horizon(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test historical VaR with multi-day horizon."""
        config = VaRConfig(
            confidence_level=0.95, horizon_days=10, method=VaRMethod.HISTORICAL
        )
        calculator = VaRCalculator(config)

        result = calculator.calculate_historical_var(sample_returns, portfolio_value)

        assert result.horizon_days == 10
        assert result.var_amount > 0

    def test_historical_var_insufficient_data(
        self, default_config: VaRConfig, portfolio_value: Decimal
    ) -> None:
        """Test historical VaR with insufficient data."""
        calculator = VaRCalculator(default_config)
        insufficient_returns = pl.Series("returns", [0.01], dtype=pl.Float64)

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_historical_var(insufficient_returns, portfolio_value)
        assert "insufficient" in str(exc_info.value).lower()

    def test_historical_var_empty_returns(
        self, default_config: VaRConfig, portfolio_value: Decimal
    ) -> None:
        """Test historical VaR with empty returns."""
        calculator = VaRCalculator(default_config)
        empty_returns = pl.Series("returns", [], dtype=pl.Float64)

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_historical_var(empty_returns, portfolio_value)
        assert "insufficient" in str(exc_info.value).lower()

    def test_historical_var_constant_returns(
        self, default_config: VaRConfig, portfolio_value: Decimal
    ) -> None:
        """Test historical VaR with constant returns (zero volatility)."""
        calculator = VaRCalculator(default_config)
        constant_returns = pl.Series("returns", [0.0] * 50, dtype=pl.Float64)

        result = calculator.calculate_historical_var(constant_returns, portfolio_value)

        # With zero volatility, VaR should be zero or near zero
        assert result.var_amount >= 0
        assert result.var_pct >= 0

    def test_historical_var_volatile_returns(
        self, volatile_prices: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test historical VaR with highly volatile returns."""
        config = VaRConfig(confidence_level=0.95, method=VaRMethod.HISTORICAL)
        calculator = VaRCalculator(config)
        returns = calculator.calculate_returns(volatile_prices)

        result = calculator.calculate_historical_var(returns, portfolio_value)

        # Volatile returns should produce higher VaR
        assert result.var_amount > 0
        assert result.expected_shortfall > result.var_amount


# ============================================================================
# Parametric VaR Tests
# ============================================================================


class TestParametricVaR:
    """Tests for parametric VaR calculation."""

    def test_parametric_var_normal_returns(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test parametric VaR with normal returns."""
        config = VaRConfig(confidence_level=0.95, method=VaRMethod.PARAMETRIC)
        calculator = VaRCalculator(config)

        result = calculator.calculate_parametric_var(sample_returns, portfolio_value)

        assert isinstance(result, VaRResult)
        assert result.var_amount > 0
        assert result.var_pct > 0
        assert result.confidence_level == 0.95
        assert result.method == VaRMethod.PARAMETRIC
        assert result.expected_shortfall >= result.var_amount

    def test_parametric_var_different_confidence_levels(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test that parametric VaR increases with confidence level."""
        config_95 = VaRConfig(confidence_level=0.95, method=VaRMethod.PARAMETRIC)
        config_99 = VaRConfig(confidence_level=0.99, method=VaRMethod.PARAMETRIC)

        calculator_95 = VaRCalculator(config_95)
        calculator_99 = VaRCalculator(config_99)

        result_95 = calculator_95.calculate_parametric_var(sample_returns, portfolio_value)
        result_99 = calculator_99.calculate_parametric_var(sample_returns, portfolio_value)

        # 99% VaR should be higher than 95% VaR
        assert result_99.var_amount > result_95.var_amount

    def test_parametric_var_multi_day_horizon(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test parametric VaR with multi-day horizon."""
        config = VaRConfig(
            confidence_level=0.95, horizon_days=10, method=VaRMethod.PARAMETRIC
        )
        calculator = VaRCalculator(config)

        result = calculator.calculate_parametric_var(sample_returns, portfolio_value)

        assert result.horizon_days == 10
        # Multi-day VaR should scale with sqrt(horizon)
        assert result.var_amount > 0

    def test_parametric_var_insufficient_data(
        self, default_config: VaRConfig, portfolio_value: Decimal
    ) -> None:
        """Test parametric VaR with insufficient data."""
        config = VaRConfig(method=VaRMethod.PARAMETRIC)
        calculator = VaRCalculator(config)
        insufficient_returns = pl.Series("returns", [0.01], dtype=pl.Float64)

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_parametric_var(insufficient_returns, portfolio_value)
        assert "insufficient" in str(exc_info.value).lower()

    def test_parametric_var_constant_returns(
        self, default_config: VaRConfig, portfolio_value: Decimal
    ) -> None:
        """Test parametric VaR with constant returns (zero volatility)."""
        config = VaRConfig(method=VaRMethod.PARAMETRIC)
        calculator = VaRCalculator(config)
        constant_returns = pl.Series("returns", [0.0] * 50, dtype=pl.Float64)

        # Should handle zero volatility gracefully
        result = calculator.calculate_parametric_var(constant_returns, portfolio_value)

        assert result.var_amount >= 0
        assert result.var_pct >= 0

    def test_parametric_var_volatile_returns(
        self, volatile_prices: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test parametric VaR with highly volatile returns."""
        config = VaRConfig(confidence_level=0.95, method=VaRMethod.PARAMETRIC)
        calculator = VaRCalculator(config)
        returns = calculator.calculate_returns(volatile_prices)

        result = calculator.calculate_parametric_var(returns, portfolio_value)

        # Volatile returns should produce higher VaR
        assert result.var_amount > 0
        assert result.expected_shortfall > result.var_amount


# ============================================================================
# Monte Carlo VaR Tests
# ============================================================================


class TestMonteCarloVaR:
    """Tests for Monte Carlo VaR calculation."""

    def test_monte_carlo_var_normal_returns(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test Monte Carlo VaR with normal returns."""
        config = VaRConfig(confidence_level=0.95, method=VaRMethod.MONTE_CARLO)
        calculator = VaRCalculator(config)

        result = calculator.calculate_monte_carlo_var(
            sample_returns, portfolio_value, simulations=10000
        )

        assert isinstance(result, VaRResult)
        assert result.var_amount > 0
        assert result.var_pct > 0
        assert result.confidence_level == 0.95
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.expected_shortfall >= result.var_amount

    def test_monte_carlo_var_different_confidence_levels(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test that Monte Carlo VaR increases with confidence level."""
        config_95 = VaRConfig(confidence_level=0.95, method=VaRMethod.MONTE_CARLO)
        config_99 = VaRConfig(confidence_level=0.99, method=VaRMethod.MONTE_CARLO)

        calculator_95 = VaRCalculator(config_95)
        calculator_99 = VaRCalculator(config_99)

        result_95 = calculator_95.calculate_monte_carlo_var(
            sample_returns, portfolio_value, simulations=10000
        )
        result_99 = calculator_99.calculate_monte_carlo_var(
            sample_returns, portfolio_value, simulations=10000
        )

        # 99% VaR should be higher than 95% VaR
        assert result_99.var_amount > result_95.var_amount

    def test_monte_carlo_var_different_simulations(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test Monte Carlo VaR with different simulation counts."""
        config = VaRConfig(confidence_level=0.95, method=VaRMethod.MONTE_CARLO)
        calculator = VaRCalculator(config)

        result_1k = calculator.calculate_monte_carlo_var(
            sample_returns, portfolio_value, simulations=1000
        )
        result_10k = calculator.calculate_monte_carlo_var(
            sample_returns, portfolio_value, simulations=10000
        )

        # Both should be valid (may differ slightly due to randomness)
        assert result_1k.var_amount > 0
        assert result_10k.var_amount > 0

    def test_monte_carlo_var_insufficient_simulations(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test Monte Carlo VaR with insufficient simulations."""
        config = VaRConfig(method=VaRMethod.MONTE_CARLO)
        calculator = VaRCalculator(config)

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_monte_carlo_var(sample_returns, portfolio_value, simulations=50)
        assert "simulation" in str(exc_info.value).lower()

    def test_monte_carlo_var_insufficient_data(
        self, default_config: VaRConfig, portfolio_value: Decimal
    ) -> None:
        """Test Monte Carlo VaR with insufficient historical data."""
        config = VaRConfig(method=VaRMethod.MONTE_CARLO)
        calculator = VaRCalculator(config)
        insufficient_returns = pl.Series("returns", [0.01], dtype=pl.Float64)

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_monte_carlo_var(
                insufficient_returns, portfolio_value, simulations=10000
            )
        assert "insufficient" in str(exc_info.value).lower()

    def test_monte_carlo_var_constant_returns(
        self, default_config: VaRConfig, portfolio_value: Decimal
    ) -> None:
        """Test Monte Carlo VaR with constant returns."""
        config = VaRConfig(method=VaRMethod.MONTE_CARLO)
        calculator = VaRCalculator(config)
        constant_returns = pl.Series("returns", [0.0] * 50, dtype=pl.Float64)

        result = calculator.calculate_monte_carlo_var(
            constant_returns, portfolio_value, simulations=10000
        )

        # Should handle zero volatility gracefully
        assert result.var_amount >= 0
        assert result.var_pct >= 0

    def test_monte_carlo_var_multi_day_horizon(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test Monte Carlo VaR with multi-day horizon."""
        config = VaRConfig(
            confidence_level=0.95, horizon_days=10, method=VaRMethod.MONTE_CARLO
        )
        calculator = VaRCalculator(config)

        result = calculator.calculate_monte_carlo_var(
            sample_returns, portfolio_value, simulations=10000
        )

        assert result.horizon_days == 10
        assert result.var_amount > 0


# ============================================================================
# Portfolio VaR Tests
# ============================================================================


class TestPortfolioVaR:
    """Tests for portfolio VaR calculation."""

    def test_portfolio_var_single_position(self, default_config: VaRConfig) -> None:
        """Test portfolio VaR with single position."""
        calculator = VaRCalculator(default_config)

        positions = [PositionVaR(symbol="AAPL", weight=1.0, var_individual=Decimal("50000"))]

        # Single position needs 1x1 correlation matrix
        correlations = pl.DataFrame([[1.0]], schema=["AAPL"])

        result = calculator.calculate_portfolio_var(positions, correlations)

        assert result.var_amount == Decimal("50000")
        assert result.confidence_level == 0.95

    def test_portfolio_var_two_positions_perfect_correlation(
        self, default_config: VaRConfig
    ) -> None:
        """Test portfolio VaR with two perfectly correlated positions."""
        calculator = VaRCalculator(default_config)

        positions = [
            PositionVaR(symbol="AAPL", weight=0.5, var_individual=Decimal("30000")),
            PositionVaR(symbol="MSFT", weight=0.5, var_individual=Decimal("40000")),
        ]

        # Perfect correlation
        correlations = pl.DataFrame([[1.0, 1.0], [1.0, 1.0]])

        result = calculator.calculate_portfolio_var(positions, correlations)

        # With perfect correlation, portfolio VaR should be weighted average
        expected_var = 0.5 * 30000 + 0.5 * 40000
        assert float(result.var_amount) == pytest.approx(expected_var, rel=1e-2)

    def test_portfolio_var_two_positions_no_correlation(
        self, default_config: VaRConfig
    ) -> None:
        """Test portfolio VaR with uncorrelated positions."""
        calculator = VaRCalculator(default_config)

        positions = [
            PositionVaR(symbol="AAPL", weight=0.5, var_individual=Decimal("30000")),
            PositionVaR(symbol="MSFT", weight=0.5, var_individual=Decimal("40000")),
        ]

        # No correlation
        correlations = pl.DataFrame([[1.0, 0.0], [0.0, 1.0]])

        result = calculator.calculate_portfolio_var(positions, correlations)

        # With no correlation, portfolio VaR should be less than weighted average
        weighted_avg = 0.5 * 30000 + 0.5 * 40000
        assert float(result.var_amount) < weighted_avg

    def test_portfolio_var_three_positions(self, default_config: VaRConfig) -> None:
        """Test portfolio VaR with three positions."""
        calculator = VaRCalculator(default_config)

        positions = [
            PositionVaR(symbol="AAPL", weight=0.5, var_individual=Decimal("30000")),
            PositionVaR(symbol="MSFT", weight=0.3, var_individual=Decimal("20000")),
            PositionVaR(symbol="GOOGL", weight=0.2, var_individual=Decimal("15000")),
        ]

        # Correlation matrix
        correlations = pl.DataFrame(
            [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]]
        )

        result = calculator.calculate_portfolio_var(positions, correlations)

        assert result.var_amount > 0
        assert result.confidence_level == 0.95

    def test_portfolio_var_no_positions(self, default_config: VaRConfig) -> None:
        """Test portfolio VaR with no positions."""
        calculator = VaRCalculator(default_config)
        correlations = pl.DataFrame()

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_portfolio_var([], correlations)
        assert "position" in str(exc_info.value).lower()

    def test_portfolio_var_weights_not_sum_to_one(self, default_config: VaRConfig) -> None:
        """Test portfolio VaR with weights not summing to 1."""
        calculator = VaRCalculator(default_config)

        positions = [
            PositionVaR(symbol="AAPL", weight=0.5, var_individual=Decimal("30000")),
            PositionVaR(symbol="MSFT", weight=0.3, var_individual=Decimal("40000")),
        ]

        correlations = pl.DataFrame([[1.0, 0.5], [0.5, 1.0]])

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_portfolio_var(positions, correlations)
        assert "weight" in str(exc_info.value).lower()

    def test_portfolio_var_mismatched_correlation_matrix(
        self, default_config: VaRConfig
    ) -> None:
        """Test portfolio VaR with mismatched correlation matrix."""
        calculator = VaRCalculator(default_config)

        positions = [
            PositionVaR(symbol="AAPL", weight=0.5, var_individual=Decimal("30000")),
            PositionVaR(symbol="MSFT", weight=0.5, var_individual=Decimal("40000")),
        ]

        # Wrong size correlation matrix (3x3 instead of 2x2)
        correlations = pl.DataFrame([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])

        with pytest.raises(ValidationError) as exc_info:
            calculator.calculate_portfolio_var(positions, correlations)
        assert "dimension" in str(exc_info.value).lower()

    def test_portfolio_var_negative_correlation(self, default_config: VaRConfig) -> None:
        """Test portfolio VaR with negative correlation (hedging)."""
        calculator = VaRCalculator(default_config)

        positions = [
            PositionVaR(symbol="AAPL", weight=0.6, var_individual=Decimal("30000")),
            PositionVaR(symbol="HEDGE", weight=0.4, var_individual=Decimal("20000")),
        ]

        # Negative correlation (hedging)
        correlations = pl.DataFrame([[1.0, -0.8], [-0.8, 1.0]])

        result = calculator.calculate_portfolio_var(positions, correlations)

        # With negative correlation, portfolio VaR should be significantly reduced
        weighted_avg = 0.6 * 30000 + 0.4 * 20000
        assert float(result.var_amount) < weighted_avg


# ============================================================================
# Calculate Method Tests
# ============================================================================


class TestCalculateMethod:
    """Tests for the unified calculate method."""

    def test_calculate_historical_method(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test calculate method with historical VaR."""
        config = VaRConfig(method=VaRMethod.HISTORICAL)
        calculator = VaRCalculator(config)

        result = calculator.calculate(sample_returns, portfolio_value)

        assert result.method == VaRMethod.HISTORICAL
        assert result.var_amount > 0

    def test_calculate_parametric_method(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test calculate method with parametric VaR."""
        config = VaRConfig(method=VaRMethod.PARAMETRIC)
        calculator = VaRCalculator(config)

        result = calculator.calculate(sample_returns, portfolio_value)

        assert result.method == VaRMethod.PARAMETRIC
        assert result.var_amount > 0

    def test_calculate_monte_carlo_method(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test calculate method with Monte Carlo VaR."""
        config = VaRConfig(method=VaRMethod.MONTE_CARLO)
        calculator = VaRCalculator(config)

        result = calculator.calculate(sample_returns, portfolio_value)

        assert result.method == VaRMethod.MONTE_CARLO
        assert result.var_amount > 0

    def test_calculate_method_comparison(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test that all three methods produce reasonable VaR estimates."""
        historical_calc = VaRCalculator(VaRConfig(method=VaRMethod.HISTORICAL))
        parametric_calc = VaRCalculator(VaRConfig(method=VaRMethod.PARAMETRIC))
        monte_carlo_calc = VaRCalculator(VaRConfig(method=VaRMethod.MONTE_CARLO))

        hist_result = historical_calc.calculate(sample_returns, portfolio_value)
        param_result = parametric_calc.calculate(sample_returns, portfolio_value)
        mc_result = monte_carlo_calc.calculate(sample_returns, portfolio_value)

        # All methods should produce positive VaR
        assert hist_result.var_amount > 0
        assert param_result.var_amount > 0
        assert mc_result.var_amount > 0

        # All methods should be within same order of magnitude
        min_var = min(
            float(hist_result.var_amount),
            float(param_result.var_amount),
            float(mc_result.var_amount),
        )
        max_var = max(
            float(hist_result.var_amount),
            float(param_result.var_amount),
            float(mc_result.var_amount),
        )

        # Ratio should be reasonable (within 5x)
        assert max_var / min_var < 5.0


# ============================================================================
# Expected Shortfall Tests
# ============================================================================


class TestExpectedShortfall:
    """Tests for Expected Shortfall calculation."""

    def test_expected_shortfall_greater_than_var(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test that Expected Shortfall is greater than or equal to VaR."""
        config = VaRConfig(confidence_level=0.95, method=VaRMethod.HISTORICAL)
        calculator = VaRCalculator(config)

        result = calculator.calculate_historical_var(sample_returns, portfolio_value)

        # ES should be >= VaR
        assert result.expected_shortfall >= result.var_amount

    def test_expected_shortfall_all_methods(
        self, sample_returns: pl.Series, portfolio_value: Decimal
    ) -> None:
        """Test that all VaR methods calculate Expected Shortfall."""
        methods = [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.MONTE_CARLO]

        for method in methods:
            config = VaRConfig(method=method)
            calculator = VaRCalculator(config)
            result = calculator.calculate(sample_returns, portfolio_value)

            assert result.expected_shortfall > 0
            assert result.expected_shortfall >= result.var_amount
