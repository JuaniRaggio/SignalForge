"""Database routing for read/write splitting with replica support.

This module provides intelligent database routing to distribute load across
primary and replica databases, improving read scalability and performance.
"""

import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from signalforge.core.logging import LoggerMixin


class DatabaseRouter(LoggerMixin):
    """Routes database queries to primary or replica databases.

    Implements read/write splitting pattern where:
    - Write operations always go to primary
    - Read operations can be distributed across replicas
    - Automatic failover to primary if all replicas are unavailable
    """

    def __init__(
        self,
        primary_url: str,
        replica_urls: list[str] | None = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
        echo: bool = False,
    ) -> None:
        """Initialize database router with primary and replica connections.

        Args:
            primary_url: Database URL for primary (read-write) database
            replica_urls: List of database URLs for read replicas
            pool_size: Number of connections to keep in pool
            max_overflow: Maximum number of connections to create beyond pool_size
            pool_pre_ping: Test connections before using them
            echo: Whether to log SQL statements
        """
        self._primary_url = primary_url
        self._replica_urls = replica_urls or []
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_pre_ping = pool_pre_ping
        self._echo = echo

        # SQLite doesn't support pool_size/max_overflow with StaticPool
        is_sqlite = "sqlite" in primary_url.lower()

        if is_sqlite:
            from sqlalchemy.pool import StaticPool

            # Create primary engine for SQLite
            self._primary_engine = create_async_engine(
                primary_url,
                echo=echo,
                poolclass=StaticPool,
            )

            # Create replica engines for SQLite
            self._replica_engines: list[AsyncEngine] = [
                create_async_engine(url, echo=echo, poolclass=StaticPool)
                for url in self._replica_urls
            ]
        else:
            # Create primary engine for PostgreSQL/other databases
            self._primary_engine = create_async_engine(
                primary_url,
                echo=echo,
                pool_pre_ping=pool_pre_ping,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )

            # Create replica engines
            self._replica_engines = [
                create_async_engine(
                    url,
                    echo=echo,
                    pool_pre_ping=pool_pre_ping,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                )
                for url in self._replica_urls
            ]

        # Track unhealthy replicas
        self._unhealthy_replicas: set[int] = set()

        self.logger.info(
            "database_router_initialized",
            primary_url=primary_url,
            replica_count=len(self._replica_engines),
        )

    def get_session_maker(
        self,
        read_only: bool = False,
        replica_preference: Literal["random", "round_robin"] = "random",
    ) -> async_sessionmaker[AsyncSession]:
        """Get a session maker for the appropriate database.

        Args:
            read_only: If True, route to replica (if available)
            replica_preference: Strategy for selecting replica

        Returns:
            Session maker configured for the selected engine
        """
        if read_only and self._replica_engines:
            engine = self._select_replica(replica_preference)
        else:
            engine = self._primary_engine

        return async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    @asynccontextmanager
    async def get_session(
        self,
        read_only: bool = False,
        replica_preference: Literal["random", "round_robin"] = "random",
    ) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic transaction management.

        Args:
            read_only: If True, route to replica (if available)
            replica_preference: Strategy for selecting replica

        Yields:
            Database session
        """
        session_maker = self.get_session_maker(read_only, replica_preference)
        async with session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def _select_replica(
        self,
        preference: Literal["random", "round_robin"] = "random",
    ) -> AsyncEngine:
        """Select a healthy replica engine.

        Args:
            preference: Selection strategy

        Returns:
            Selected replica engine, or primary if no healthy replicas
        """
        # Get list of healthy replicas
        healthy_replicas = [
            idx
            for idx in range(len(self._replica_engines))
            if idx not in self._unhealthy_replicas
        ]

        if not healthy_replicas:
            self.logger.warning(
                "no_healthy_replicas",
                total_replicas=len(self._replica_engines),
                unhealthy_count=len(self._unhealthy_replicas),
            )
            return self._primary_engine

        if preference == "random":
            selected_idx = random.choice(healthy_replicas)
        else:
            # Round-robin: select first healthy replica
            selected_idx = healthy_replicas[0]

        return self._replica_engines[selected_idx]

    async def check_replica_health(self, replica_idx: int) -> bool:
        """Check if a specific replica is healthy.

        Args:
            replica_idx: Index of replica to check

        Returns:
            True if replica is healthy
        """
        if replica_idx >= len(self._replica_engines):
            return False

        try:
            engine = self._replica_engines[replica_idx]
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            # Remove from unhealthy set if it was there
            self._unhealthy_replicas.discard(replica_idx)
            return True
        except Exception as e:
            self.logger.warning(
                "replica_health_check_failed",
                replica_idx=replica_idx,
                error=str(e),
            )
            self._unhealthy_replicas.add(replica_idx)
            return False

    async def check_all_replicas(self) -> dict[int, bool]:
        """Check health of all replicas.

        Returns:
            Dictionary mapping replica index to health status
        """
        results = {}
        for idx in range(len(self._replica_engines)):
            results[idx] = await self.check_replica_health(idx)

        healthy_count = sum(1 for status in results.values() if status)
        self.logger.info(
            "replica_health_check_completed",
            total_replicas=len(self._replica_engines),
            healthy_count=healthy_count,
        )

        return results

    async def check_primary_health(self) -> bool:
        """Check if primary database is healthy.

        Returns:
            True if primary is healthy
        """
        try:
            async with self._primary_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(
                "primary_health_check_failed",
                error=str(e),
            )
            return False

    async def get_replication_lag(self, replica_idx: int) -> float | None:
        """Get replication lag for a specific replica in seconds.

        Args:
            replica_idx: Index of replica to check

        Returns:
            Replication lag in seconds, or None if unable to determine
        """
        if replica_idx >= len(self._replica_engines):
            return None

        try:
            engine = self._replica_engines[replica_idx]
            async with engine.connect() as conn:
                # PostgreSQL-specific query for replication lag
                result = await conn.execute(
                    text(
                        """
                        SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
                        AS lag_seconds
                        """
                    )
                )
                row = result.fetchone()
                if row:
                    return float(row[0]) if row[0] is not None else 0.0
                return None
        except Exception as e:
            self.logger.warning(
                "replication_lag_check_failed",
                replica_idx=replica_idx,
                error=str(e),
            )
            return None

    async def get_all_replication_lags(self) -> dict[int, float | None]:
        """Get replication lag for all replicas.

        Returns:
            Dictionary mapping replica index to lag in seconds
        """
        results = {}
        for idx in range(len(self._replica_engines)):
            results[idx] = await self.get_replication_lag(idx)
        return results

    async def close(self) -> None:
        """Close all database connections."""
        self.logger.info("closing_database_connections")

        await self._primary_engine.dispose()
        for engine in self._replica_engines:
            await engine.dispose()

        self.logger.info("database_connections_closed")

    @property
    def replica_count(self) -> int:
        """Get total number of replicas."""
        return len(self._replica_engines)

    @property
    def healthy_replica_count(self) -> int:
        """Get number of healthy replicas."""
        return len(self._replica_engines) - len(self._unhealthy_replicas)

    @property
    def has_replicas(self) -> bool:
        """Check if any replicas are configured."""
        return len(self._replica_engines) > 0
