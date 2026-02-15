"""Tests for volume filter module."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from signalforge.execution.schemas import VolumeFilterResult
from signalforge.execution.volume_filter import (
    VolumeFilter,
    VolumeFilterConfig,
    VolumeRejectionReason,
    calculate_adv,
)


class TestVolumeFilterConfig:
    """Tests for VolumeFilterConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VolumeFilterConfig()

        assert config.min_adv_shares == 100_000
        assert config.min_adv_dollars == 1_000_000.0
        assert config.max_position_adv_ratio == 0.01
        assert config.adv_window == 20

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VolumeFilterConfig(
            min_adv_shares=500_000,
            min_adv_dollars=5_000_000.0,
            max_position_adv_ratio=0.02,
            adv_window=30,
        )

        assert config.min_adv_shares == 500_000
        assert config.min_adv_dollars == 5_000_000.0
        assert config.max_position_adv_ratio == 0.02
        assert config.adv_window == 30

    def test_negative_min_adv_shares_raises_error(self) -> None:
        """Test that negative min_adv_shares raises ValidationError."""
        with pytest.raises(ValidationError):
            VolumeFilterConfig(min_adv_shares=-1000)

    def test_negative_min_adv_dollars_raises_error(self) -> None:
        """Test that negative min_adv_dollars raises ValidationError."""
        with pytest.raises(ValidationError):
            VolumeFilterConfig(min_adv_dollars=-1000.0)

    def test_zero_max_position_ratio_raises_error(self) -> None:
        """Test that max_position_adv_ratio <= 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            VolumeFilterConfig(max_position_adv_ratio=0.0)

    def test_max_position_ratio_greater_than_one_raises_error(self) -> None:
        """Test that max_position_adv_ratio > 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            VolumeFilterConfig(max_position_adv_ratio=1.5)

    def test_negative_max_position_ratio_raises_error(self) -> None:
        """Test that negative max_position_adv_ratio raises ValidationError."""
        with pytest.raises(ValidationError):
            VolumeFilterConfig(max_position_adv_ratio=-0.01)

    def test_zero_adv_window_raises_error(self) -> None:
        """Test that adv_window < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            VolumeFilterConfig(adv_window=0)

    def test_negative_adv_window_raises_error(self) -> None:
        """Test that negative adv_window raises ValidationError."""
        with pytest.raises(ValidationError):
            VolumeFilterConfig(adv_window=-10)

    def test_boundary_max_position_ratio_valid(self) -> None:
        """Test that max_position_adv_ratio = 1.0 is valid."""
        config = VolumeFilterConfig(max_position_adv_ratio=1.0)
        assert config.max_position_adv_ratio == 1.0


class TestVolumeFilterResult:
    """Tests for VolumeFilterResult model."""

    def test_passing_result(self) -> None:
        """Test creating a passing result."""
        result = VolumeFilterResult(
            symbol="AAPL",
            passes_filter=True,
            position_pct_of_adv=0.5,
            max_position_size=100000.0,
            reason=None,
        )

        assert result.symbol == "AAPL"
        assert result.passes_filter is True
        assert result.position_pct_of_adv == 0.5
        assert result.max_position_size == 100000.0
        assert result.reason is None

    def test_rejected_result(self) -> None:
        """Test creating a rejected result."""
        result = VolumeFilterResult(
            symbol="XYZ",
            passes_filter=False,
            position_pct_of_adv=2.5,
            max_position_size=10000.0,
            reason="Position exceeds limit",
        )

        assert result.symbol == "XYZ"
        assert result.passes_filter is False
        assert result.position_pct_of_adv == 2.5
        assert result.max_position_size == 10000.0
        assert result.reason == "Position exceeds limit"


class TestVolumeRejectionReason:
    """Tests for VolumeRejectionReason enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert VolumeRejectionReason.BELOW_MIN_ADV == "below_min_adv"
        assert VolumeRejectionReason.EXCEEDS_POSITION_LIMIT == "exceeds_position_limit"
        assert VolumeRejectionReason.INSUFFICIENT_LIQUIDITY == "insufficient_liquidity"


class TestCalculateAdv:
    """Tests for calculate_adv function."""

    def test_basic_calculation(self) -> None:
        """Test basic ADV calculation."""
        df = pl.DataFrame({
            "volume": [100, 200, 300, 400, 500] * 4,
            "close": [10.0] * 20,
        })

        result = calculate_adv(df, window=5)

        assert "adv_shares" in result.columns
        assert "adv_dollars" in result.columns
        # First 4 rows will be null (window - 1)
        assert result["adv_shares"][4] == 300.0  # Average of [100, 200, 300, 400, 500]

    def test_window_smaller_than_data(self) -> None:
        """Test with window smaller than data length."""
        df = pl.DataFrame({
            "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "close": [50.0] * 10,
        })

        result = calculate_adv(df, window=3)

        assert "adv_shares" in result.columns
        # Window of 3, so first 2 values are null
        assert result["adv_shares"][2] == 200.0  # Average of [100, 200, 300]

    def test_adv_dollars_calculation(self) -> None:
        """Test ADV dollars is calculated correctly."""
        df = pl.DataFrame({
            "volume": [100, 200, 300],
            "close": [10.0, 20.0, 30.0],
        })

        result = calculate_adv(df, window=2)

        # ADV dollars = ADV shares * close price
        assert "adv_dollars" in result.columns

    def test_without_close_column(self) -> None:
        """Test that ADV shares works without close column."""
        df = pl.DataFrame({
            "volume": [100, 200, 300, 400, 500],
        })

        result = calculate_adv(df, window=3)

        assert "adv_shares" in result.columns
        assert "adv_dollars" not in result.columns


class TestVolumeFilter:
    """Tests for VolumeFilter class."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        vf = VolumeFilter()

        assert vf.max_position_pct_of_adv == 0.01
        assert vf.min_adv_threshold == 100_000

    def test_initialization_custom(self) -> None:
        """Test custom initialization."""
        vf = VolumeFilter(
            max_position_pct_of_adv=0.05,
            min_adv_threshold=500_000,
        )

        assert vf.max_position_pct_of_adv == 0.05
        assert vf.min_adv_threshold == 500_000

    def test_initialization_invalid_max_position(self) -> None:
        """Test invalid max_position_pct_of_adv raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            VolumeFilter(max_position_pct_of_adv=0.0)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            VolumeFilter(max_position_pct_of_adv=1.5)

    def test_initialization_invalid_min_adv(self) -> None:
        """Test invalid min_adv_threshold raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            VolumeFilter(min_adv_threshold=-100)

    def test_check_passes_all(self) -> None:
        """Test check passes when all constraints are met."""
        vf = VolumeFilter(
            max_position_pct_of_adv=0.01,
            min_adv_threshold=100_000,
        )

        # 1M ADV * $100 = $100M ADV dollars
        # Position: 100 shares * $100 = $10,000
        # Max position: $100M * 0.01 = $1M
        result = vf.check(
            symbol="AAPL",
            position_size_shares=100,
            adv_20=1_000_000,  # 1M shares
            price=100.0,
        )

        assert result.passes_filter is True
        assert result.symbol == "AAPL"
        assert result.reason is None

    def test_check_fails_low_adv(self) -> None:
        """Test check fails when ADV is below threshold."""
        vf = VolumeFilter(
            max_position_pct_of_adv=0.01,
            min_adv_threshold=1_000_000,  # $1M minimum
        )

        # ADV: 10,000 shares * $10 = $100K (below $1M threshold)
        result = vf.check(
            symbol="XYZ",
            position_size_shares=100,
            adv_20=10_000,
            price=10.0,
        )

        assert result.passes_filter is False
        assert "below minimum threshold" in result.reason

    def test_check_fails_position_too_large(self) -> None:
        """Test check fails when position exceeds limit."""
        vf = VolumeFilter(
            max_position_pct_of_adv=0.01,  # 1%
            min_adv_threshold=100_000,
        )

        # ADV: 100,000 shares * $100 = $10M
        # Max position: $10M * 0.01 = $100K
        # Requested position: 2,000 shares * $100 = $200K (exceeds limit)
        result = vf.check(
            symbol="TEST",
            position_size_shares=2_000,
            adv_20=100_000,
            price=100.0,
        )

        assert result.passes_filter is False
        assert "exceeds max" in result.reason

    def test_get_max_position(self) -> None:
        """Test get_max_position calculation."""
        vf = VolumeFilter(max_position_pct_of_adv=0.05)

        # ADV: 100,000 shares * $50 = $5M
        # Max: $5M * 0.05 = $250K
        max_pos = vf.get_max_position(adv_20=100_000, price=50.0)

        assert max_pos == 250_000.0

    def test_get_max_position_different_ratios(self) -> None:
        """Test get_max_position with different ratios."""
        vf1 = VolumeFilter(max_position_pct_of_adv=0.01)
        vf2 = VolumeFilter(max_position_pct_of_adv=0.10)

        # ADV: 1M shares * $100 = $100M
        max_pos1 = vf1.get_max_position(adv_20=1_000_000, price=100.0)
        max_pos2 = vf2.get_max_position(adv_20=1_000_000, price=100.0)

        assert max_pos1 == 1_000_000.0  # 1% of $100M
        assert max_pos2 == 10_000_000.0  # 10% of $100M


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_adv(self) -> None:
        """Test handling of zero ADV."""
        vf = VolumeFilter(min_adv_threshold=100_000)

        result = vf.check(
            symbol="ZERO",
            position_size_shares=100,
            adv_20=0,
            price=100.0,
        )

        assert result.passes_filter is False

    def test_very_small_position(self) -> None:
        """Test with very small position size."""
        vf = VolumeFilter()

        result = vf.check(
            symbol="SMALL",
            position_size_shares=1,
            adv_20=1_000_000,
            price=100.0,
        )

        assert result.passes_filter is True

    def test_position_at_exact_limit(self) -> None:
        """Test position at exactly the limit."""
        vf = VolumeFilter(
            max_position_pct_of_adv=0.01,
            min_adv_threshold=0,
        )

        # ADV: 100,000 shares * $100 = $10M
        # Max position: $10M * 0.01 = $100K
        # Position: 1,000 shares * $100 = $100K (exactly at limit)
        result = vf.check(
            symbol="EXACT",
            position_size_shares=1_000,
            adv_20=100_000,
            price=100.0,
        )

        assert result.passes_filter is True

    def test_position_just_over_limit(self) -> None:
        """Test position just over the limit."""
        vf = VolumeFilter(
            max_position_pct_of_adv=0.01,
            min_adv_threshold=0,
        )

        # ADV: 100,000 shares * $100 = $10M
        # Max position: $10M * 0.01 = $100K
        # Position: 1,001 shares * $100 = $100.1K (just over limit)
        result = vf.check(
            symbol="OVER",
            position_size_shares=1_001,
            adv_20=100_000,
            price=100.0,
        )

        assert result.passes_filter is False

    def test_high_price_low_volume(self) -> None:
        """Test stock with high price but low volume."""
        vf = VolumeFilter(min_adv_threshold=1_000_000)

        # ADV: 100 shares * $10,000 = $1M (at threshold)
        result = vf.check(
            symbol="BRK",
            position_size_shares=1,
            adv_20=100,
            price=10_000.0,
        )

        assert result.passes_filter is True

    def test_low_price_high_volume(self) -> None:
        """Test penny stock with high volume."""
        vf = VolumeFilter(min_adv_threshold=100_000)

        # ADV: 100,000,000 shares * $0.01 = $1M
        result = vf.check(
            symbol="PENNY",
            position_size_shares=1_000,
            adv_20=100_000_000,
            price=0.01,
        )

        assert result.passes_filter is True

    def test_conservative_position_limit(self) -> None:
        """Test with very conservative position limit."""
        vf = VolumeFilter(max_position_pct_of_adv=0.001)  # 0.1%

        max_pos = vf.get_max_position(adv_20=1_000_000, price=100.0)

        assert max_pos == 100_000.0  # 0.1% of $100M

    def test_aggressive_position_limit(self) -> None:
        """Test with aggressive position limit."""
        vf = VolumeFilter(max_position_pct_of_adv=1.0)  # 100%

        max_pos = vf.get_max_position(adv_20=1_000_000, price=100.0)

        assert max_pos == 100_000_000.0  # 100% of $100M
