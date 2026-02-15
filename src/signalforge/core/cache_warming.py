"""Cache warming strategies for improved application startup and performance.

This module provides intelligent cache warming to pre-populate Redis cache
with frequently accessed data, reducing initial response times and improving
overall system performance.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from signalforge.core.logging import LoggerMixin


@dataclass
class CacheEntry:
    """Represents a cache entry to be warmed."""

    key: str
    value: Any
    ttl_seconds: int | None = None


@dataclass
class WarmingStats:
    """Statistics from a cache warming operation."""

    total_keys: int
    successful_keys: int
    failed_keys: int
    execution_time_seconds: float
    timestamp: datetime


class CacheWarmer(LoggerMixin):
    """Manages cache warming strategies for Redis.

    Features:
    - Pre-load frequently accessed data at startup
    - Periodic refresh of cached data
    - Intelligent invalidation strategies
    - Warming statistics and monitoring
    """

    def __init__(
        self,
        redis_client: Redis,
        engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize cache warmer.

        Args:
            redis_client: Redis client for cache operations
            engine: Optional SQLAlchemy engine for database queries
        """
        self._redis = redis_client
        self._engine = engine
        self._warming_strategies: dict[str, Callable[[], Awaitable[list[CacheEntry]]]] = {}

    def register_strategy(
        self,
        name: str,
        strategy: Callable[[], Awaitable[list[CacheEntry]]],
    ) -> None:
        """Register a cache warming strategy.

        Args:
            name: Name of the strategy
            strategy: Async callable that returns list of CacheEntry objects
        """
        self._warming_strategies[name] = strategy
        self.logger.info(
            "cache_warming_strategy_registered",
            strategy_name=name,
        )

    async def warm_cache(
        self,
        strategies: list[str] | None = None,
        parallel: bool = True,
    ) -> WarmingStats:
        """Execute cache warming strategies.

        Args:
            strategies: List of strategy names to execute (None = all)
            parallel: Whether to execute strategies in parallel

        Returns:
            Statistics from the warming operation
        """
        start_time = datetime.now(UTC)

        # Determine which strategies to run
        if strategies is None:
            strategies_to_run = list(self._warming_strategies.keys())
        else:
            strategies_to_run = [s for s in strategies if s in self._warming_strategies]

        self.logger.info(
            "cache_warming_started",
            strategies=strategies_to_run,
            parallel=parallel,
        )

        total_keys = 0
        successful_keys = 0
        failed_keys = 0

        if parallel:
            # Execute strategies in parallel
            tasks = [
                self._execute_strategy(name)
                for name in strategies_to_run
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(
                        "cache_warming_strategy_failed",
                        error=str(result),
                    )
                    failed_keys += 1
                elif isinstance(result, tuple):
                    total_keys += result[0]
                    successful_keys += result[1]
                    failed_keys += result[2]
        else:
            # Execute strategies sequentially
            for name in strategies_to_run:
                try:
                    t, s, f = await self._execute_strategy(name)
                    total_keys += t
                    successful_keys += s
                    failed_keys += f
                except Exception as e:
                    self.logger.error(
                        "cache_warming_strategy_failed",
                        strategy_name=name,
                        error=str(e),
                    )
                    failed_keys += 1

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        stats = WarmingStats(
            total_keys=total_keys,
            successful_keys=successful_keys,
            failed_keys=failed_keys,
            execution_time_seconds=execution_time,
            timestamp=datetime.now(UTC),
        )

        self.logger.info(
            "cache_warming_completed",
            total_keys=total_keys,
            successful_keys=successful_keys,
            failed_keys=failed_keys,
            execution_time_seconds=execution_time,
        )

        return stats

    async def _execute_strategy(
        self,
        strategy_name: str,
    ) -> tuple[int, int, int]:
        """Execute a single cache warming strategy.

        Args:
            strategy_name: Name of the strategy to execute

        Returns:
            Tuple of (total_keys, successful_keys, failed_keys)
        """
        strategy = self._warming_strategies[strategy_name]

        try:
            # Execute strategy to get cache entries
            entries = await strategy()

            if not isinstance(entries, list):
                self.logger.error(
                    "invalid_strategy_return_type",
                    strategy_name=strategy_name,
                    expected="list[CacheEntry]",
                )
                return (0, 0, 1)

            # Populate cache with entries
            total = len(entries)
            successful = 0
            failed = 0

            for entry in entries:
                try:
                    if entry.ttl_seconds:
                        await self._redis.setex(
                            entry.key,
                            entry.ttl_seconds,
                            entry.value,
                        )
                    else:
                        await self._redis.set(entry.key, entry.value)
                    successful += 1
                except Exception as e:
                    self.logger.warning(
                        "cache_entry_set_failed",
                        key=entry.key,
                        error=str(e),
                    )
                    failed += 1

            self.logger.info(
                "strategy_executed",
                strategy_name=strategy_name,
                total_keys=total,
                successful_keys=successful,
                failed_keys=failed,
            )

            return (total, successful, failed)
        except Exception as e:
            self.logger.error(
                "strategy_execution_failed",
                strategy_name=strategy_name,
                error=str(e),
            )
            return (0, 0, 1)

    async def warm_popular_tickers(
        self,
        limit: int = 100,
        ttl_seconds: int = 3600,
    ) -> list[CacheEntry]:
        """Warm cache with most popular stock tickers.

        Args:
            limit: Number of tickers to cache
            ttl_seconds: Cache TTL in seconds

        Returns:
            List of cache entries
        """
        if not self._engine:
            return []

        try:
            # Query most popular tickers from database
            query = """
            SELECT DISTINCT symbol, name
            FROM market_data
            ORDER BY volume DESC
            LIMIT :limit
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query), {"limit": limit})
                rows = result.fetchall()

            entries = [
                CacheEntry(
                    key=f"ticker:{row[0]}",
                    value=f"{row[0]}:{row[1]}",
                    ttl_seconds=ttl_seconds,
                )
                for row in rows
            ]

            self.logger.info(
                "popular_tickers_warmed",
                count=len(entries),
            )

            return entries
        except Exception as e:
            self.logger.error(
                "warm_popular_tickers_failed",
                error=str(e),
            )
            return []

    async def warm_recent_prices(
        self,
        symbols: list[str],
        ttl_seconds: int = 300,
    ) -> list[CacheEntry]:
        """Warm cache with recent price data for specific symbols.

        Args:
            symbols: List of stock symbols
            ttl_seconds: Cache TTL in seconds

        Returns:
            List of cache entries
        """
        if not self._engine or not symbols:
            return []

        try:
            # Build query for latest prices
            symbol_list = ", ".join(f"'{s}'" for s in symbols)
            query = f"""
            SELECT symbol, close, volume, timestamp
            FROM (
                SELECT symbol, close, volume, timestamp,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                FROM market_data
                WHERE symbol IN ({symbol_list})
            ) t
            WHERE rn = 1
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                rows = result.fetchall()

            entries = [
                CacheEntry(
                    key=f"price:{row[0]}",
                    value=f"{row[1]}:{row[2]}:{row[3]}",
                    ttl_seconds=ttl_seconds,
                )
                for row in rows
            ]

            self.logger.info(
                "recent_prices_warmed",
                count=len(entries),
            )

            return entries
        except Exception as e:
            self.logger.error(
                "warm_recent_prices_failed",
                error=str(e),
            )
            return []

    async def warm_user_preferences(
        self,
        ttl_seconds: int = 1800,
    ) -> list[CacheEntry]:
        """Warm cache with user preferences.

        Args:
            ttl_seconds: Cache TTL in seconds

        Returns:
            List of cache entries
        """
        if not self._engine:
            return []

        try:
            # Query active users' preferences
            query = """
            SELECT user_id, preferences
            FROM user_preferences
            WHERE last_active > NOW() - INTERVAL '7 days'
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                rows = result.fetchall()

            entries = [
                CacheEntry(
                    key=f"user_pref:{row[0]}",
                    value=str(row[1]),
                    ttl_seconds=ttl_seconds,
                )
                for row in rows
            ]

            self.logger.info(
                "user_preferences_warmed",
                count=len(entries),
            )

            return entries
        except Exception as e:
            self.logger.error(
                "warm_user_preferences_failed",
                error=str(e),
            )
            return []

    async def invalidate_pattern(
        self,
        pattern: str,
    ) -> int:
        """Invalidate all cache keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "ticker:*")

        Returns:
            Number of keys invalidated
        """
        try:
            # Scan for matching keys
            cursor = 0
            keys_deleted = 0

            while True:
                cursor, keys = await self._redis.scan(
                    cursor,
                    match=pattern,
                    count=100,
                )

                if keys:
                    deleted = await self._redis.delete(*keys)
                    keys_deleted += deleted

                if cursor == 0:
                    break

            self.logger.info(
                "cache_pattern_invalidated",
                pattern=pattern,
                keys_deleted=keys_deleted,
            )

            return keys_deleted
        except Exception as e:
            self.logger.error(
                "cache_invalidation_failed",
                pattern=pattern,
                error=str(e),
            )
            return 0

    async def refresh_cache_entry(
        self,
        key: str,
        fetch_func: Callable[[], Awaitable[Any]],
        ttl_seconds: int | None = None,
    ) -> bool:
        """Refresh a specific cache entry.

        Args:
            key: Cache key to refresh
            fetch_func: Async callable to fetch fresh data
            ttl_seconds: Optional TTL for the refreshed entry

        Returns:
            True if refresh was successful
        """
        try:
            # Fetch fresh data
            value = await fetch_func()

            # Update cache
            if ttl_seconds:
                await self._redis.setex(key, ttl_seconds, value)
            else:
                await self._redis.set(key, value)

            self.logger.debug(
                "cache_entry_refreshed",
                key=key,
            )

            return True
        except Exception as e:
            self.logger.error(
                "cache_refresh_failed",
                key=key,
                error=str(e),
            )
            return False

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        try:
            info = await self._redis.info("stats")
            keyspace = await self._redis.info("keyspace")

            return {
                "total_keys": sum(
                    int(v.get("keys", 0))
                    for v in keyspace.values()
                    if isinstance(v, dict)
                ),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                ) * 100,
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0),
            }
        except Exception as e:
            self.logger.error(
                "get_cache_stats_failed",
                error=str(e),
            )
            return {}
