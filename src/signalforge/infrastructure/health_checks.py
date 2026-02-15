"""Advanced health checks for infrastructure components.

This module provides comprehensive health checking capabilities for
all infrastructure components including databases, caches, and external services.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from signalforge.core.logging import LoggerMixin


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] | None = None
    timestamp: datetime | None = None


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: dict[str, ComponentHealth]
    timestamp: datetime


class AdvancedHealthChecker(LoggerMixin):
    """Comprehensive health checker for all system components.

    Features:
    - Database connectivity and latency checks
    - Redis connectivity and latency checks
    - TimescaleDB-specific health checks
    - Replication lag monitoring
    - Connection pool health
    - Chunk health monitoring
    """

    def __init__(
        self,
        database_engine: AsyncEngine,
        redis_client: Redis,
        replica_engines: list[AsyncEngine] | None = None,
    ) -> None:
        """Initialize advanced health checker.

        Args:
            database_engine: Primary database engine
            redis_client: Redis client
            replica_engines: Optional list of replica database engines
        """
        self._db_engine = database_engine
        self._redis = redis_client
        self._replica_engines = replica_engines or []

        # Health check thresholds
        self._latency_warning_ms: float = 100.0
        self._latency_error_ms: float = 500.0
        self._replication_lag_warning_seconds: float = 10.0
        self._replication_lag_error_seconds: float = 60.0

    async def check_database(self) -> ComponentHealth:
        """Check primary database health.

        Returns:
            Database health status
        """
        start_time = datetime.now(UTC)

        try:
            async with self._db_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Determine status based on latency
            if latency_ms < self._latency_warning_ms:
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"
            elif latency_ms < self._latency_error_ms:
                status = HealthStatus.DEGRADED
                message = f"Database latency elevated: {latency_ms:.2f}ms"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Database latency critical: {latency_ms:.2f}ms"

            return ComponentHealth(
                name="database",
                status=status,
                latency_ms=latency_ms,
                message=message,
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            self.logger.error(
                "database_health_check_failed",
                error=str(e),
            )
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.now(UTC),
            )

    async def check_redis(self) -> ComponentHealth:
        """Check Redis health.

        Returns:
            Redis health status
        """
        start_time = datetime.now(UTC)

        try:
            await self._redis.ping()

            latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Determine status based on latency
            if latency_ms < self._latency_warning_ms:
                status = HealthStatus.HEALTHY
                message = "Redis connection healthy"
            elif latency_ms < self._latency_error_ms:
                status = HealthStatus.DEGRADED
                message = f"Redis latency elevated: {latency_ms:.2f}ms"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Redis latency critical: {latency_ms:.2f}ms"

            # Get additional Redis stats
            info = await self._redis.info("memory")
            details = {
                "used_memory_human": info.get("used_memory_human"),
                "maxmemory_human": info.get("maxmemory_human"),
            }

            return ComponentHealth(
                name="redis",
                status=status,
                latency_ms=latency_ms,
                message=message,
                details=details,
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            self.logger.error(
                "redis_health_check_failed",
                error=str(e),
            )
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                timestamp=datetime.now(UTC),
            )

    async def check_timescaledb(self) -> ComponentHealth:
        """Check TimescaleDB extension health.

        Returns:
            TimescaleDB health status
        """
        try:
            async with self._db_engine.connect() as conn:
                # Check if extension is available
                result = await conn.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
                )
                row = result.fetchone()

                if row is None:
                    return ComponentHealth(
                        name="timescaledb",
                        status=HealthStatus.UNHEALTHY,
                        message="TimescaleDB extension not installed",
                        timestamp=datetime.now(UTC),
                    )

                # Get TimescaleDB version
                result = await conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                )
                version_row = result.fetchone()
                version = version_row[0] if version_row else "unknown"

                return ComponentHealth(
                    name="timescaledb",
                    status=HealthStatus.HEALTHY,
                    message="TimescaleDB extension available",
                    details={"version": version},
                    timestamp=datetime.now(UTC),
                )
        except Exception as e:
            self.logger.error(
                "timescaledb_health_check_failed",
                error=str(e),
            )
            return ComponentHealth(
                name="timescaledb",
                status=HealthStatus.UNHEALTHY,
                message=f"TimescaleDB check failed: {str(e)}",
                timestamp=datetime.now(UTC),
            )

    async def check_replication_lag(self) -> ComponentHealth:
        """Check replication lag for all replicas.

        Returns:
            Replication lag health status
        """
        if not self._replica_engines:
            return ComponentHealth(
                name="replication",
                status=HealthStatus.HEALTHY,
                message="No replicas configured",
                timestamp=datetime.now(UTC),
            )

        try:
            max_lag_seconds = 0.0
            replica_lags: list[dict[str, Any]] = []

            for idx, engine in enumerate(self._replica_engines):
                try:
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
                        lag_seconds = float(row[0]) if row and row[0] else 0.0
                        replica_lags.append({"replica_idx": idx, "lag_seconds": lag_seconds})
                        max_lag_seconds = max(max_lag_seconds, lag_seconds)
                except Exception as e:
                    self.logger.warning(
                        "replica_lag_check_failed",
                        replica_idx=idx,
                        error=str(e),
                    )
                    replica_lags.append({"replica_idx": idx, "error": str(e)})

            # Determine status based on max lag
            if max_lag_seconds < self._replication_lag_warning_seconds:
                status = HealthStatus.HEALTHY
                message = f"Replication lag healthy: {max_lag_seconds:.2f}s"
            elif max_lag_seconds < self._replication_lag_error_seconds:
                status = HealthStatus.DEGRADED
                message = f"Replication lag elevated: {max_lag_seconds:.2f}s"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Replication lag critical: {max_lag_seconds:.2f}s"

            return ComponentHealth(
                name="replication",
                status=status,
                message=message,
                details={
                    "max_lag_seconds": max_lag_seconds,
                    "replicas": replica_lags,
                },
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            self.logger.error(
                "replication_lag_check_failed",
                error=str(e),
            )
            return ComponentHealth(
                name="replication",
                status=HealthStatus.UNHEALTHY,
                message=f"Replication check failed: {str(e)}",
                timestamp=datetime.now(UTC),
            )

    async def check_chunk_health(
        self,
        table_name: str,
    ) -> ComponentHealth:
        """Check TimescaleDB chunk health for a specific table.

        Args:
            table_name: Name of the hypertable to check

        Returns:
            Chunk health status
        """
        try:
            query = f"""
            SELECT
                COUNT(*) as total_chunks,
                SUM(CASE WHEN is_compressed THEN 1 ELSE 0 END) as compressed_chunks,
                SUM(total_bytes) as total_bytes,
                pg_size_pretty(SUM(total_bytes)) as total_size
            FROM timescaledb_information.chunks
            WHERE hypertable_name = '{table_name}'
            """

            async with self._db_engine.connect() as conn:
                result = await conn.execute(text(query))
                row = result.fetchone()

                if row:
                    total_chunks = row[0]
                    compressed_chunks = row[1] or 0
                    total_bytes = row[2] or 0
                    total_size = row[3]

                    compression_ratio = (
                        (compressed_chunks / total_chunks * 100) if total_chunks > 0 else 0
                    )

                    return ComponentHealth(
                        name=f"chunks_{table_name}",
                        status=HealthStatus.HEALTHY,
                        message=f"Table {table_name} chunks healthy",
                        details={
                            "total_chunks": total_chunks,
                            "compressed_chunks": compressed_chunks,
                            "compression_ratio_percent": compression_ratio,
                            "total_size": total_size,
                            "total_bytes": total_bytes,
                        },
                        timestamp=datetime.now(UTC),
                    )
                else:
                    return ComponentHealth(
                        name=f"chunks_{table_name}",
                        status=HealthStatus.DEGRADED,
                        message=f"No chunks found for table {table_name}",
                        timestamp=datetime.now(UTC),
                    )
        except Exception as e:
            self.logger.error(
                "chunk_health_check_failed",
                table_name=table_name,
                error=str(e),
            )
            return ComponentHealth(
                name=f"chunks_{table_name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Chunk check failed: {str(e)}",
                timestamp=datetime.now(UTC),
            )

    async def check_connection_pool(self) -> ComponentHealth:
        """Check database connection pool health.

        Returns:
            Connection pool health status
        """
        try:
            pool = self._db_engine.pool

            size = pool.size()  # type: ignore[attr-defined]
            checked_in = pool.checkedin()  # type: ignore[attr-defined]
            checked_out = pool.checkedout()  # type: ignore[attr-defined]
            overflow = pool.overflow()  # type: ignore[attr-defined]

            utilization = (checked_out / size * 100) if size > 0 else 0

            # Determine status based on utilization
            if utilization < 70:
                status = HealthStatus.HEALTHY
                message = f"Connection pool healthy ({utilization:.1f}% utilization)"
            elif utilization < 90:
                status = HealthStatus.DEGRADED
                message = f"Connection pool under pressure ({utilization:.1f}% utilization)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Connection pool saturated ({utilization:.1f}% utilization)"

            return ComponentHealth(
                name="connection_pool",
                status=status,
                message=message,
                details={
                    "size": size,
                    "checked_in": checked_in,
                    "checked_out": checked_out,
                    "overflow": overflow,
                    "utilization_percent": utilization,
                },
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            self.logger.error(
                "connection_pool_health_check_failed",
                error=str(e),
            )
            return ComponentHealth(
                name="connection_pool",
                status=HealthStatus.UNHEALTHY,
                message=f"Connection pool check failed: {str(e)}",
                timestamp=datetime.now(UTC),
            )

    async def check_all(
        self,
        include_chunks: list[str] | None = None,
    ) -> SystemHealth:
        """Perform all health checks.

        Args:
            include_chunks: Optional list of table names to check chunk health

        Returns:
            Overall system health status
        """
        # Run all checks in parallel
        tasks = [
            self.check_database(),
            self.check_redis(),
            self.check_timescaledb(),
            self.check_replication_lag(),
            self.check_connection_pool(),
        ]

        # Add chunk health checks if requested
        if include_chunks:
            for table_name in include_chunks:
                tasks.append(self.check_chunk_health(table_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build components dictionary
        components: dict[str, ComponentHealth] = {}
        for result in results:
            if isinstance(result, ComponentHealth):
                components[result.name] = result
            elif isinstance(result, Exception):
                self.logger.error(
                    "health_check_failed",
                    error=str(result),
                )

        # Determine overall status
        statuses = [c.status for c in components.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        system_health = SystemHealth(
            status=overall_status,
            components=components,
            timestamp=datetime.now(UTC),
        )

        self.logger.info(
            "health_check_completed",
            status=overall_status.value,
            components_count=len(components),
        )

        return system_health

    def set_latency_thresholds(
        self,
        warning_ms: float,
        error_ms: float,
    ) -> None:
        """Set latency thresholds for health checks.

        Args:
            warning_ms: Warning threshold in milliseconds
            error_ms: Error threshold in milliseconds
        """
        self._latency_warning_ms = warning_ms
        self._latency_error_ms = error_ms

    def set_replication_lag_thresholds(
        self,
        warning_seconds: float,
        error_seconds: float,
    ) -> None:
        """Set replication lag thresholds.

        Args:
            warning_seconds: Warning threshold in seconds
            error_seconds: Error threshold in seconds
        """
        self._replication_lag_warning_seconds = warning_seconds
        self._replication_lag_error_seconds = error_seconds
