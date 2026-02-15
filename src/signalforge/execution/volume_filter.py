"""Volume-based filtering for signal executability.

This module filters trading signals based on volume constraints to ensure
positions can be entered and exited without excessive market impact.
"""

from __future__ import annotations

from enum import Enum

import polars as pl
from pydantic import BaseModel, Field

from signalforge.core.logging import get_logger
from signalforge.execution.schemas import VolumeFilterResult

logger = get_logger(__name__)


class VolumeRejectionReason(str, Enum):
    """Reasons for rejecting a trade based on volume."""

    BELOW_MIN_ADV = "below_min_adv"
    EXCEEDS_POSITION_LIMIT = "exceeds_position_limit"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"


class VolumeFilterConfig(BaseModel):
    """Configuration for volume filtering."""

    min_adv_shares: int = Field(default=100_000, ge=0)
    min_adv_dollars: float = Field(default=1_000_000.0, ge=0)
    max_position_adv_ratio: float = Field(default=0.01, gt=0, le=1)
    adv_window: int = Field(default=20, ge=1)


def calculate_adv(
    df: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """Calculate Average Daily Volume.

    Args:
        df: DataFrame with 'volume' and optionally 'close' columns.
        window: Rolling window size for averaging.

    Returns:
        DataFrame with ADV columns added.
    """
    result = df.with_columns(
        pl.col("volume").rolling_mean(window_size=window).alias("adv_shares")
    )

    if "close" in df.columns:
        result = result.with_columns(
            (pl.col("adv_shares") * pl.col("close")).alias("adv_dollars")
        )

    return result


class VolumeFilter:
    """Filter signals based on volume constraints.

    Ensures that position sizes are reasonable relative to average daily
    volume to avoid excessive market impact and slippage.
    """

    def __init__(
        self,
        max_position_pct_of_adv: float = 0.01,  # 1% of ADV
        min_adv_threshold: float = 100_000,  # Minimum $100K ADV
    ) -> None:
        """Initialize volume filter with constraints.

        Args:
            max_position_pct_of_adv: Maximum position size as fraction of ADV (default: 0.01 = 1%).
            min_adv_threshold: Minimum average daily volume threshold in dollars (default: $100K).

        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0 < max_position_pct_of_adv <= 1:
            raise ValueError("max_position_pct_of_adv must be between 0 and 1")
        if min_adv_threshold < 0:
            raise ValueError("min_adv_threshold cannot be negative")

        self.max_position_pct_of_adv = max_position_pct_of_adv
        self.min_adv_threshold = min_adv_threshold

        logger.debug(
            "volume_filter_initialized",
            max_position_pct_of_adv=max_position_pct_of_adv,
            min_adv_threshold=min_adv_threshold,
        )

    def check(
        self,
        symbol: str,
        position_size_shares: int,
        adv_20: float,
        price: float,
    ) -> VolumeFilterResult:
        """Check if position passes volume filter.

        Args:
            symbol: Trading symbol.
            position_size_shares: Desired position size in shares.
            adv_20: Average daily volume over 20 days in shares.
            price: Current share price.

        Returns:
            VolumeFilterResult with pass/fail status and details.
        """
        logger.info(
            "checking_volume_filter",
            symbol=symbol,
            position_size_shares=position_size_shares,
            adv_20=adv_20,
            price=price,
        )

        # Calculate ADV in dollars
        adv_dollars = adv_20 * price

        # Calculate position size in dollars
        position_size_dollars = position_size_shares * price

        # Check minimum ADV threshold
        if adv_dollars < self.min_adv_threshold:
            logger.warning(
                "volume_filter_failed_min_adv",
                symbol=symbol,
                adv_dollars=adv_dollars,
                min_threshold=self.min_adv_threshold,
            )
            return VolumeFilterResult(
                symbol=symbol,
                passes_filter=False,
                position_pct_of_adv=0.0,
                max_position_size=0.0,
                reason=f"ADV ${adv_dollars:,.0f} below minimum threshold ${self.min_adv_threshold:,.0f}",
            )

        # Calculate position as percentage of ADV
        position_pct_of_adv = position_size_dollars / adv_dollars if adv_dollars > 0 else 0

        # Calculate max allowable position
        max_position_size = self.get_max_position(adv_20, price)

        # Check if position exceeds limit
        if position_size_dollars > max_position_size:
            logger.warning(
                "volume_filter_failed_max_position",
                symbol=symbol,
                position_size_dollars=position_size_dollars,
                max_position_size=max_position_size,
                position_pct_of_adv=position_pct_of_adv,
            )
            return VolumeFilterResult(
                symbol=symbol,
                passes_filter=False,
                position_pct_of_adv=position_pct_of_adv * 100,  # Convert to percentage
                max_position_size=max_position_size,
                reason=f"Position ${position_size_dollars:,.0f} exceeds max ${max_position_size:,.0f} ({self.max_position_pct_of_adv*100:.1f}% of ADV)",
            )

        # Passed all checks
        logger.info(
            "volume_filter_passed",
            symbol=symbol,
            position_size_dollars=position_size_dollars,
            position_pct_of_adv=position_pct_of_adv,
            max_position_size=max_position_size,
        )

        return VolumeFilterResult(
            symbol=symbol,
            passes_filter=True,
            position_pct_of_adv=position_pct_of_adv * 100,  # Convert to percentage
            max_position_size=max_position_size,
            reason=None,
        )

    def get_max_position(
        self,
        adv_20: float,
        price: float,
    ) -> float:
        """Calculate maximum allowable position size.

        Args:
            adv_20: Average daily volume over 20 days in shares.
            price: Current share price.

        Returns:
            Maximum position size in dollars.
        """
        adv_dollars = adv_20 * price
        max_position = adv_dollars * self.max_position_pct_of_adv
        return max_position
