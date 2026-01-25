"""Liquidity assessment for trading signals.

This module provides comprehensive liquidity analysis for assets, enabling
traders to assess whether signals are executable under real market conditions.

Key Features:
- Average Daily Volume (ADV) calculation
- Volume volatility measurement
- Liquidity scoring system (0-100)
- Redis caching for performance optimization

Examples:
    Basic liquidity assessment:

    >>> import polars as pl
    >>> from signalforge.execution.liquidity import assess_liquidity
    >>>
    >>> df = pl.DataFrame({
    ...     "timestamp": [...],
    ...     "close": [...],
    ...     "volume": [...],
    ... })
    >>> metrics = assess_liquidity(df, "AAPL")
    >>> if metrics.is_liquid:
    ...     print(f"Asset is liquid with score: {metrics.liquidity_score}")

    Using cached metrics:

    >>> from signalforge.execution.liquidity import get_cached_liquidity_metrics
    >>> metrics = await get_cached_liquidity_metrics(df, "AAPL", redis_client)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# Liquidity scoring thresholds
HIGH_LIQUIDITY_THRESHOLD = 70.0
MEDIUM_LIQUIDITY_THRESHOLD = 40.0

# Cache configuration
LIQUIDITY_CACHE_TTL = 3600  # 1 hour in seconds


@dataclass
class LiquidityMetrics:
    """Metrics for assessing asset liquidity.

    Attributes:
        symbol: Asset symbol being analyzed.
        avg_daily_volume: Average Daily Volume over the specified window (default 20 days).
        volume_volatility: Standard deviation of volume, indicating consistency.
        liquidity_score: Composite score from 0-100 indicating overall liquidity.
        is_liquid: Boolean flag indicating if the asset meets liquidity threshold.
    """

    symbol: str
    avg_daily_volume: float
    volume_volatility: float
    liquidity_score: float
    is_liquid: bool

    def __post_init__(self) -> None:
        """Validate liquidity metrics."""
        if self.avg_daily_volume < 0:
            raise ValueError("avg_daily_volume cannot be negative")
        if self.volume_volatility < 0:
            raise ValueError("volume_volatility cannot be negative")
        if not 0 <= self.liquidity_score <= 100:
            raise ValueError("liquidity_score must be between 0 and 100")

    def to_dict(self) -> dict[str, str | float | bool]:
        """Convert metrics to dictionary for serialization."""
        return asdict(self)


def calculate_avg_daily_volume(df: pl.DataFrame, window: int = 20) -> float:
    """Calculate Average Daily Volume over window period.

    Args:
        df: DataFrame containing volume data with a 'volume' column.
        window: Number of days to calculate average over (default: 20).

    Returns:
        Average daily volume as a float.

    Raises:
        ValueError: If volume column is missing or DataFrame is empty.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"volume": [1000000, 1200000, 900000, 1100000]})
        >>> adv = calculate_avg_daily_volume(df, window=4)
        >>> print(f"ADV: {adv}")
    """
    if "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'volume' column")

    if df.height == 0:
        raise ValueError("DataFrame cannot be empty")

    # Take the last 'window' days of data
    recent_data = df.tail(window) if df.height >= window else df

    # Calculate mean volume
    avg_volume = float(recent_data["volume"].mean())

    if avg_volume is None or pl.Series([avg_volume]).is_null().any():
        logger.warning("calculate_avg_daily_volume returned null, using 0.0")
        return 0.0

    logger.debug(
        "calculated_avg_daily_volume",
        window=window,
        rows_used=recent_data.height,
        avg_volume=avg_volume,
    )

    return avg_volume


def calculate_liquidity_score(
    avg_volume: float,
    price: float,
    volume_volatility: float,
) -> float:
    """Calculate liquidity score from 0-100.

    The liquidity score is a composite metric that considers:
    - Dollar volume (volume * price): Higher is better
    - Volume consistency (lower volatility is better)

    Scoring Guidelines:
    - Score > 70: High liquidity (suitable for most trading strategies)
    - Score 40-70: Medium liquidity (acceptable with caution)
    - Score < 40: Low liquidity (risky for execution)

    Args:
        avg_volume: Average daily volume in shares/units.
        price: Current or average price of the asset.
        volume_volatility: Standard deviation of volume (lower is more consistent).

    Returns:
        Liquidity score from 0.0 to 100.0.

    Raises:
        ValueError: If inputs are negative or price is zero.

    Examples:
        >>> score = calculate_liquidity_score(
        ...     avg_volume=5_000_000,
        ...     price=150.0,
        ...     volume_volatility=500_000
        ... )
        >>> print(f"Liquidity score: {score:.2f}")
    """
    if avg_volume < 0:
        raise ValueError("avg_volume cannot be negative")
    if price <= 0:
        raise ValueError("price must be positive")
    if volume_volatility < 0:
        raise ValueError("volume_volatility cannot be negative")

    # Handle edge case of zero volume
    if avg_volume == 0:
        logger.warning("calculate_liquidity_score called with zero avg_volume")
        return 0.0

    # Calculate dollar volume (notional value traded per day)
    dollar_volume = avg_volume * price

    # Base score from dollar volume (logarithmic scale)
    # $1M daily volume -> ~50 points
    # $10M daily volume -> ~70 points
    # $100M+ daily volume -> ~90+ points
    import math

    if dollar_volume > 0:
        volume_score = min(100.0, 30.0 + 30.0 * math.log10(dollar_volume / 1_000_000))
    else:
        volume_score = 0.0

    # Consistency score from volume volatility
    # Lower volatility (more consistent) is better
    if avg_volume > 0:
        coefficient_of_variation = volume_volatility / avg_volume
    else:
        coefficient_of_variation = 1.0

    # CV < 0.2 (20%) is excellent, CV > 1.0 (100%) is poor
    consistency_score = max(0.0, 40.0 * (1.0 - min(coefficient_of_variation, 1.0)))

    # Combine scores (60% volume, 40% consistency)
    liquidity_score = 0.6 * volume_score + 0.4 * consistency_score

    # Ensure score is within bounds
    liquidity_score = max(0.0, min(100.0, liquidity_score))

    logger.debug(
        "calculated_liquidity_score",
        avg_volume=avg_volume,
        price=price,
        dollar_volume=dollar_volume,
        volume_volatility=volume_volatility,
        volume_score=volume_score,
        consistency_score=consistency_score,
        final_score=liquidity_score,
    )

    return liquidity_score


def assess_liquidity(df: pl.DataFrame, symbol: str, window: int = 20) -> LiquidityMetrics:
    """Assess liquidity for a given symbol from price data.

    This is the primary function for comprehensive liquidity assessment.
    It calculates all relevant metrics and returns a structured result.

    Args:
        df: DataFrame with columns: 'timestamp', 'close', 'volume'.
        symbol: Asset symbol being analyzed.
        window: Number of days for ADV calculation (default: 20).

    Returns:
        LiquidityMetrics object with comprehensive liquidity assessment.

    Raises:
        ValueError: If required columns are missing or data is invalid.

    Examples:
        >>> import polars as pl
        >>> from datetime import datetime, timedelta
        >>>
        >>> dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        >>> df = pl.DataFrame({
        ...     "timestamp": dates,
        ...     "close": [150.0] * 30,
        ...     "volume": [5_000_000 + i * 10_000 for i in range(30)],
        ... })
        >>>
        >>> metrics = assess_liquidity(df, "AAPL")
        >>> print(f"Liquidity Score: {metrics.liquidity_score:.2f}")
        >>> print(f"Is Liquid: {metrics.is_liquid}")
    """
    # Validate required columns
    required_cols = ["timestamp", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.height == 0:
        raise ValueError("DataFrame cannot be empty")

    logger.info(
        "assessing_liquidity",
        symbol=symbol,
        rows=df.height,
        window=window,
    )

    # Calculate average daily volume
    avg_daily_volume = calculate_avg_daily_volume(df, window)

    # Calculate volume volatility (standard deviation)
    recent_data = df.tail(window) if df.height >= window else df
    volume_std = float(recent_data["volume"].std())

    # Handle null standard deviation (can happen with constant volume)
    if volume_std is None or pl.Series([volume_std]).is_null().any():
        logger.warning("volume std returned null, using 0.0")
        volume_std = 0.0

    # Get current/average price for dollar volume calculation
    avg_price = float(recent_data["close"].mean())
    if avg_price is None or avg_price <= 0:
        logger.warning("avg_price is invalid, using 1.0 for calculation")
        avg_price = 1.0

    # Calculate liquidity score
    liquidity_score = calculate_liquidity_score(
        avg_volume=avg_daily_volume,
        price=avg_price,
        volume_volatility=volume_std,
    )

    # Determine if asset is liquid based on threshold
    is_liquid = liquidity_score > MEDIUM_LIQUIDITY_THRESHOLD

    metrics = LiquidityMetrics(
        symbol=symbol,
        avg_daily_volume=avg_daily_volume,
        volume_volatility=volume_std,
        liquidity_score=liquidity_score,
        is_liquid=is_liquid,
    )

    logger.info(
        "liquidity_assessed",
        symbol=symbol,
        avg_daily_volume=avg_daily_volume,
        liquidity_score=liquidity_score,
        is_liquid=is_liquid,
    )

    return metrics


async def get_cached_liquidity_metrics(
    df: pl.DataFrame,
    symbol: str,
    redis_client: Redis,
    window: int = 20,
    force_refresh: bool = False,
) -> LiquidityMetrics:
    """Get liquidity metrics with Redis caching.

    This function checks Redis cache first, and only calculates metrics
    if the cache is empty or expired. This significantly improves performance
    for frequently accessed symbols.

    Args:
        df: DataFrame with price and volume data.
        symbol: Asset symbol being analyzed.
        redis_client: Redis client for caching.
        window: Number of days for ADV calculation (default: 20).
        force_refresh: If True, bypass cache and recalculate (default: False).

    Returns:
        LiquidityMetrics object (from cache or freshly calculated).

    Examples:
        >>> from signalforge.core.redis import get_redis
        >>> redis = await get_redis()
        >>> metrics = await get_cached_liquidity_metrics(df, "AAPL", redis)
    """
    cache_key = f"liquidity:metrics:{symbol}:{window}"

    # Try to get from cache if not forcing refresh
    if not force_refresh:
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.debug("liquidity_metrics_cache_hit", symbol=symbol)
                metrics_dict = json.loads(cached_data)
                return LiquidityMetrics(**metrics_dict)
        except Exception as e:
            logger.warning(
                "liquidity_cache_read_failed",
                symbol=symbol,
                error=str(e),
            )

    # Cache miss or forced refresh - calculate metrics
    logger.debug("liquidity_metrics_cache_miss", symbol=symbol)
    metrics = assess_liquidity(df, symbol, window)

    # Store in cache
    try:
        metrics_json = json.dumps(metrics.to_dict())
        await redis_client.setex(cache_key, LIQUIDITY_CACHE_TTL, metrics_json)
        logger.debug(
            "liquidity_metrics_cached",
            symbol=symbol,
            ttl=LIQUIDITY_CACHE_TTL,
        )
    except Exception as e:
        logger.warning(
            "liquidity_cache_write_failed",
            symbol=symbol,
            error=str(e),
        )

    return metrics
