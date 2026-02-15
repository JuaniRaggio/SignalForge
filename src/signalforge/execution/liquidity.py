"""Liquidity scoring module for execution quality assessment.

This module provides comprehensive liquidity analysis for trading symbols,
enabling traders to assess whether signals are executable under real market conditions.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.execution.schemas import LiquidityScore
from signalforge.models.price import Price

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class LiquidityScorer:
    """Score stock liquidity for execution quality.

    Calculates a comprehensive liquidity score based on multiple factors
    including volume, spread, market capitalization, and order book depth.

    The scoring system weights each component to produce a final score
    from 0-100, with higher scores indicating better liquidity.
    """

    def __init__(
        self,
        volume_weight: float = 0.4,
        spread_weight: float = 0.3,
        market_cap_weight: float = 0.2,
        depth_weight: float = 0.1,
    ) -> None:
        """Initialize liquidity scorer with component weights.

        Args:
            volume_weight: Weight for volume-based score (default: 0.4).
            spread_weight: Weight for spread-based score (default: 0.3).
            market_cap_weight: Weight for market cap score (default: 0.2).
            depth_weight: Weight for order book depth score (default: 0.1).

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        self.volume_weight = volume_weight
        self.spread_weight = spread_weight
        self.market_cap_weight = market_cap_weight
        self.depth_weight = depth_weight

        total_weight = volume_weight + spread_weight + market_cap_weight + depth_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        logger.debug(
            "liquidity_scorer_initialized",
            volume_weight=volume_weight,
            spread_weight=spread_weight,
            market_cap_weight=market_cap_weight,
            depth_weight=depth_weight,
        )

    async def score(self, symbol: str, session: AsyncSession) -> LiquidityScore:
        """Calculate liquidity score for a symbol from database.

        Args:
            symbol: Trading symbol to score.
            session: Async database session.

        Returns:
            LiquidityScore with comprehensive assessment.

        Raises:
            ValueError: If insufficient data available.
        """
        logger.info("calculating_liquidity_score", symbol=symbol)

        # Fetch last 20 days of price data
        stmt = (
            select(Price)
            .where(Price.symbol == symbol)
            .order_by(Price.timestamp.desc())
            .limit(20)
        )
        result = await session.execute(stmt)
        prices = result.scalars().all()

        if len(prices) < 10:
            raise ValueError(f"Insufficient data for {symbol}: only {len(prices)} days available")

        # Convert to Polars for calculation
        df = pl.DataFrame(
            {
                "timestamp": [p.timestamp for p in prices],
                "close": [float(p.close) for p in prices],
                "volume": [p.volume for p in prices],
                "high": [float(p.high) for p in prices],
                "low": [float(p.low) for p in prices],
            }
        )

        # Calculate ADV
        adv_mean = df["volume"].mean()
        if adv_mean is None:
            adv_20 = 0.0
        else:
            # Cast to float, mypy doesn't understand Polars types well
            adv_20 = float(adv_mean)  # type: ignore[arg-type]

        # Calculate average spread (using high-low as proxy)
        df = df.with_columns(
            ((pl.col("high") - pl.col("low")) / pl.col("close") * 10000).alias("spread_bps")
        )
        spread_mean = df["spread_bps"].mean()
        if spread_mean is None:
            avg_spread_bps = 0.0
        else:
            # Cast to float, mypy doesn't understand Polars types well
            avg_spread_bps = float(spread_mean)  # type: ignore[arg-type]

        return self.score_from_data(
            symbol=symbol, adv_20=adv_20, avg_spread_bps=avg_spread_bps, market_cap=None
        )

    def score_from_data(
        self,
        symbol: str,
        adv_20: float,
        avg_spread_bps: float,
        market_cap: float | None = None,
    ) -> LiquidityScore:
        """Calculate score from provided data without database access.

        Args:
            symbol: Trading symbol.
            adv_20: Average daily volume over 20 days.
            avg_spread_bps: Average spread in basis points.
            market_cap: Market capitalization in dollars (optional).

        Returns:
            LiquidityScore with calculated metrics.
        """
        logger.debug(
            "scoring_from_data",
            symbol=symbol,
            adv_20=adv_20,
            avg_spread_bps=avg_spread_bps,
            market_cap=market_cap,
        )

        # Calculate component scores
        volume_score = self._calculate_volume_score(adv_20)
        spread_score = self._calculate_spread_score(avg_spread_bps)
        market_cap_score = self._calculate_market_cap_score(market_cap) if market_cap else 50.0

        # Depth score defaults to 50 if not available
        depth_score = 50.0

        # Calculate weighted total score
        total_score = (
            volume_score * self.volume_weight
            + spread_score * self.spread_weight
            + market_cap_score * self.market_cap_weight
            + depth_score * self.depth_weight
        )

        # Ensure score is within bounds
        total_score = max(0.0, min(100.0, total_score))

        rating = self._get_rating(total_score)

        liquidity_score = LiquidityScore(
            symbol=symbol,
            timestamp=datetime.now(),
            score=total_score,
            volume_score=volume_score,
            spread_score=spread_score,
            market_cap_score=market_cap_score,
            adv_20=adv_20,
            rating=rating,
        )

        logger.info(
            "liquidity_scored",
            symbol=symbol,
            score=total_score,
            rating=rating,
            adv_20=adv_20,
        )

        return liquidity_score

    def _calculate_volume_score(self, adv: float) -> float:
        """Calculate volume-based score.

        Scoring Guidelines:
        - ADV > 10M shares: 100 points (highly liquid)
        - ADV > 1M shares: 80 points (liquid)
        - ADV > 100K shares: 60 points (moderately liquid)
        - ADV > 10K shares: 40 points (low liquidity)
        - ADV <= 10K shares: 20 points (illiquid)

        Args:
            adv: Average daily volume in shares.

        Returns:
            Volume score (0-100).
        """
        if adv > 10_000_000:
            return 100.0
        elif adv > 1_000_000:
            return 80.0
        elif adv > 100_000:
            return 60.0
        elif adv > 10_000:
            return 40.0
        else:
            return 20.0

    def _calculate_spread_score(self, spread_bps: float) -> float:
        """Calculate spread-based score.

        Scoring Guidelines:
        - Spread < 5 bps: 100 points (institutional quality)
        - Spread < 10 bps: 80 points (very tight)
        - Spread < 25 bps: 60 points (acceptable)
        - Spread < 50 bps: 40 points (wide)
        - Spread >= 50 bps: 20 points (very wide)

        Args:
            spread_bps: Spread in basis points.

        Returns:
            Spread score (0-100).
        """
        if spread_bps < 5:
            return 100.0
        elif spread_bps < 10:
            return 80.0
        elif spread_bps < 25:
            return 60.0
        elif spread_bps < 50:
            return 40.0
        else:
            return 20.0

    def _calculate_market_cap_score(self, market_cap: float) -> float:
        """Calculate market capitalization score.

        Args:
            market_cap: Market cap in dollars.

        Returns:
            Market cap score (0-100).
        """
        if market_cap > 10_000_000_000:  # > $10B (large cap)
            return 100.0
        elif market_cap > 2_000_000_000:  # > $2B (mid cap)
            return 80.0
        elif market_cap > 300_000_000:  # > $300M (small cap)
            return 60.0
        elif market_cap > 50_000_000:  # > $50M (micro cap)
            return 40.0
        else:  # <= $50M (nano cap)
            return 20.0

    def _get_rating(self, score: float) -> str:
        """Convert numeric score to categorical rating.

        Args:
            score: Numeric liquidity score (0-100).

        Returns:
            Rating string.
        """
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "illiquid"
