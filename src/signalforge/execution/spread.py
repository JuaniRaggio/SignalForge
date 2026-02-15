"""Spread calculation and tracking for execution quality assessment.

This module provides bid-ask spread calculation and analysis capabilities,
allowing traders to assess execution costs and market conditions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.execution.schemas import SpreadMetrics
from signalforge.models.price import Price

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class SpreadConfig(BaseModel):
    """Configuration for spread calculation."""

    lookback_days: int = Field(default=20, ge=1)
    min_data_points: int = Field(default=5, ge=1)


def calculate_corwin_schultz_spread(
    high: float,
    low: float,
    high_prev: float,
    low_prev: float,
) -> float:
    """Calculate spread using Corwin-Schultz estimator.

    This estimator uses two-day high and low prices to estimate
    the bid-ask spread, based on the assumption that high prices
    are typically buyer-initiated and low prices seller-initiated.

    Args:
        high: Current day high price.
        low: Current day low price.
        high_prev: Previous day high price.
        low_prev: Previous day low price.

    Returns:
        Estimated spread in decimal form.
    """
    # Calculate beta
    high_2day = max(high, high_prev)
    low_2day = min(low, low_prev)

    gamma = np.log(high_2day / low_2day) ** 2
    beta = np.log(high / low) ** 2 + np.log(high_prev / low_prev) ** 2

    # Calculate alpha
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))

    # Calculate spread
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Ensure non-negative
    return max(0.0, float(spread))


class SpreadCalculator:
    """Calculate and track bid-ask spreads.

    Estimates spreads from OHLC data when tick-level bid-ask data is
    not available. Uses high-low ranges as a proxy for spread estimation.
    """

    def __init__(self, lookback_days: int = 20) -> None:
        """Initialize spread calculator.

        Args:
            lookback_days: Number of days to use for historical analysis (default: 20).

        Raises:
            ValueError: If lookback_days is less than 1.
        """
        if lookback_days < 1:
            raise ValueError("Lookback days must be at least 1")

        self.lookback_days = lookback_days

        logger.debug("spread_calculator_initialized", lookback_days=lookback_days)

    async def get_spread_metrics(
        self, symbol: str, session: AsyncSession
    ) -> SpreadMetrics:
        """Get spread metrics from database.

        Args:
            symbol: Trading symbol to analyze.
            session: Async database session.

        Returns:
            SpreadMetrics with spread analysis.

        Raises:
            ValueError: If insufficient data available.
        """
        logger.info("getting_spread_metrics", symbol=symbol, lookback_days=self.lookback_days)

        # Fetch price data
        stmt = (
            select(Price)
            .where(Price.symbol == symbol)
            .order_by(Price.timestamp.desc())
            .limit(self.lookback_days)
        )
        result = await session.execute(stmt)
        prices = result.scalars().all()

        if len(prices) < 5:
            raise ValueError(
                f"Insufficient data for {symbol}: only {len(prices)} days available"
            )

        # Convert to lists for calculation
        highs = [float(p.high) for p in prices]
        lows = [float(p.low) for p in prices]
        closes = [float(p.close) for p in prices]

        # Calculate spreads using high-low as proxy
        # bid = low (buyers), ask = high (sellers)
        spreads = []
        for high, low, _close in zip(highs, lows, closes, strict=True):
            spread_bps = self.calculate_spread_bps(low, high)
            spreads.append(spread_bps)

        # Current spread is most recent
        current_spread = spreads[0]

        return self.calculate_metrics_from_data(
            symbol=symbol, spreads=spreads, current_spread=current_spread
        )

    def calculate_spread_bps(self, bid: float, ask: float) -> float:
        """Calculate spread in basis points.

        Args:
            bid: Bid price (or low price as proxy).
            ask: Ask price (or high price as proxy).

        Returns:
            Spread in basis points.

        Raises:
            ValueError: If bid or ask are invalid.
        """
        if bid <= 0 or ask <= 0:
            raise ValueError("Bid and ask must be positive")
        if bid > ask:
            raise ValueError("Bid cannot be greater than ask")

        # Calculate midpoint
        midpoint = (bid + ask) / 2.0

        # Calculate spread as percentage of midpoint
        spread_pct = (ask - bid) / midpoint

        # Convert to basis points
        spread_bps = spread_pct * 10000

        return spread_bps

    def calculate_metrics_from_data(
        self,
        symbol: str,
        spreads: list[float],
        current_spread: float,
    ) -> SpreadMetrics:
        """Calculate metrics from spread history.

        Args:
            symbol: Trading symbol.
            spreads: Historical spread values in basis points.
            current_spread: Current spread in basis points.

        Returns:
            SpreadMetrics with analysis.

        Raises:
            ValueError: If spreads list is empty.
        """
        if not spreads:
            raise ValueError("Spreads list cannot be empty")

        logger.debug(
            "calculating_spread_metrics",
            symbol=symbol,
            num_spreads=len(spreads),
            current_spread=current_spread,
        )

        # Calculate average spread
        avg_spread_20d_bps = float(np.mean(spreads))

        # Calculate spread volatility
        spread_volatility = float(np.std(spreads))

        # Calculate percentile rank of current spread
        # Lower percentile means current spread is tighter than usual
        percentile_rank = float(
            sum(1 for s in spreads if s <= current_spread) / len(spreads) * 100
        )

        metrics = SpreadMetrics(
            symbol=symbol,
            current_spread_bps=current_spread,
            avg_spread_20d_bps=avg_spread_20d_bps,
            spread_volatility=spread_volatility,
            percentile_rank=percentile_rank,
        )

        logger.info(
            "spread_metrics_calculated",
            symbol=symbol,
            current_spread_bps=current_spread,
            avg_spread_20d_bps=avg_spread_20d_bps,
            percentile_rank=percentile_rank,
        )

        return metrics
