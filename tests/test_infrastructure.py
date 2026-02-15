"""Comprehensive tests for infrastructure components."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from signalforge.core.cache_warming import CacheEntry, CacheWarmer, WarmingStats
from signalforge.core.compression import CompressionPolicy
from signalforge.core.database_routing import DatabaseRouter
from signalforge.core.retention import RetentionPolicy, RetentionStats, UserTier
from signalforge.infrastructure.connection_pool import (
    ConnectionPoolManager,
    PoolMetrics,
)
from signalforge.infrastructure.health_checks import (
    AdvancedHealthChecker,
    ComponentHealth,
    HealthStatus,
)
from signalforge.infrastructure.scaling_policies import (
    MetricType,
    ScalingDirection,
    ScalingMetrics,
    ScalingPolicy,
    ScalingThreshold,
)


class TestDatabaseRouter:
    """Tests for DatabaseRouter."""

    @pytest_asyncio.fixture
    async def primary_engine(self) -> AsyncEngine:
        """Create primary test engine."""
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
        )

    @pytest_asyncio.fixture
    async def replica_engine(self) -> AsyncEngine:
        """Create replica test engine."""
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
        )

    @pytest_asyncio.fixture
    async def router(self, primary_engine: AsyncEngine) -> DatabaseRouter:
        """Create database router instance."""
        return DatabaseRouter(
            primary_url="sqlite+aiosqlite:///:memory:",
            replica_urls=["sqlite+aiosqlite:///:memory:"],
        )

    @pytest.mark.asyncio
    async def test_router_initialization(self, router: DatabaseRouter) -> None:
        """Test router initialization."""
        assert router.has_replicas is True
        assert router.replica_count == 1
        assert router.healthy_replica_count == 1

    @pytest.mark.asyncio
    async def test_get_session_primary(self, router: DatabaseRouter) -> None:
        """Test getting session for primary database."""
        async with router.get_session(read_only=False) as session:
            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_get_session_replica(self, router: DatabaseRouter) -> None:
        """Test getting session for replica database."""
        async with router.get_session(read_only=True) as session:
            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_check_primary_health(self, router: DatabaseRouter) -> None:
        """Test primary database health check."""
        is_healthy = await router.check_primary_health()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_check_replica_health(self, router: DatabaseRouter) -> None:
        """Test replica health check."""
        is_healthy = await router.check_replica_health(0)
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_check_all_replicas(self, router: DatabaseRouter) -> None:
        """Test checking all replicas."""
        results = await router.check_all_replicas()
        assert len(results) == 1
        assert results[0] is True


class TestConnectionPoolManager:
    """Tests for ConnectionPoolManager."""

    @pytest_asyncio.fixture
    async def pool_manager(self) -> ConnectionPoolManager:
        """Create connection pool manager instance."""
        manager = ConnectionPoolManager(
            database_url="sqlite+aiosqlite:///:memory:",
            min_pool_size=2,
            max_pool_size=10,
        )
        await manager.initialize()
        yield manager
        await manager.dispose()

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool_manager: ConnectionPoolManager) -> None:
        """Test pool initialization."""
        assert pool_manager.is_initialized is True
        assert pool_manager.is_shutting_down is False

    @pytest.mark.asyncio
    async def test_get_current_metrics(self, pool_manager: ConnectionPoolManager) -> None:
        """Test getting current pool metrics."""
        metrics = pool_manager.get_current_metrics()
        assert isinstance(metrics, PoolMetrics)
        assert metrics.size >= 0
        assert metrics.utilization_percent >= 0

    @pytest.mark.asyncio
    async def test_health_check(self, pool_manager: ConnectionPoolManager) -> None:
        """Test pool health check."""
        is_healthy = await pool_manager.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_adjust_pool_size(self, pool_manager: ConnectionPoolManager) -> None:
        """Test adjusting pool size."""
        success = await pool_manager.adjust_pool_size(5)
        assert success is True

    @pytest.mark.asyncio
    async def test_get_metrics_summary(self, pool_manager: ConnectionPoolManager) -> None:
        """Test getting metrics summary."""
        # Collect some metrics first
        pool_manager.get_current_metrics()
        summary = pool_manager.get_metrics_summary(lookback_minutes=5)
        assert "sample_count" in summary


class TestCompressionPolicy:
    """Tests for CompressionPolicy."""

    @pytest_asyncio.fixture
    async def engine(self) -> AsyncEngine:
        """Create test engine."""
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
        )

    @pytest_asyncio.fixture
    async def compression_policy(self, engine: AsyncEngine) -> CompressionPolicy:
        """Create compression policy instance."""
        return CompressionPolicy(engine)

    @pytest.mark.asyncio
    async def test_policy_initialization(
        self, compression_policy: CompressionPolicy
    ) -> None:
        """Test compression policy initialization."""
        assert compression_policy is not None

    @pytest.mark.asyncio
    async def test_get_all_compression_policies(
        self, compression_policy: CompressionPolicy
    ) -> None:
        """Test getting all compression policies."""
        policies = await compression_policy.get_all_compression_policies()
        assert isinstance(policies, list)


class TestRetentionPolicy:
    """Tests for RetentionPolicy."""

    @pytest_asyncio.fixture
    async def engine(self) -> AsyncEngine:
        """Create test engine."""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
        )
        # Create a test table
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE test_data (
                        id INTEGER PRIMARY KEY,
                        timestamp DATETIME,
                        value TEXT
                    )
                    """
                )
            )
        yield engine
        await engine.dispose()

    @pytest_asyncio.fixture
    async def retention_policy(self, engine: AsyncEngine) -> RetentionPolicy:
        """Create retention policy instance."""
        return RetentionPolicy(engine)

    @pytest.mark.asyncio
    async def test_policy_initialization(
        self, retention_policy: RetentionPolicy
    ) -> None:
        """Test retention policy initialization."""
        assert retention_policy is not None

    @pytest.mark.asyncio
    async def test_get_tier_retention_days(
        self, retention_policy: RetentionPolicy
    ) -> None:
        """Test getting retention days for tier."""
        days = retention_policy.get_tier_retention_days(UserTier.FREE)
        assert days == 30

    @pytest.mark.asyncio
    async def test_get_retention_policies(
        self, retention_policy: RetentionPolicy
    ) -> None:
        """Test getting retention policies."""
        policies = await retention_policy.get_retention_policies()
        assert isinstance(policies, list)

    @pytest.mark.asyncio
    async def test_manual_retention_cleanup(
        self, retention_policy: RetentionPolicy, engine: AsyncEngine
    ) -> None:
        """Test manual retention cleanup."""
        # Insert test data
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO test_data (timestamp, value)
                    VALUES (:timestamp, :value)
                    """
                ),
                {
                    "timestamp": datetime.now(UTC) - timedelta(days=60),
                    "value": "old",
                },
            )

        stats = await retention_policy.manual_retention_cleanup(
            table_name="test_data",
            time_column="timestamp",
            retention_days=30,
        )

        assert isinstance(stats, RetentionStats)
        assert stats.rows_deleted >= 0

    @pytest.mark.asyncio
    async def test_get_data_age_stats(
        self, retention_policy: RetentionPolicy, engine: AsyncEngine
    ) -> None:
        """Test getting data age statistics."""
        # Insert test data
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO test_data (timestamp, value)
                    VALUES (:timestamp, :value)
                    """
                ),
                {
                    "timestamp": datetime.now(UTC),
                    "value": "test",
                },
            )

        stats = await retention_policy.get_data_age_stats(
            table_name="test_data",
            time_column="timestamp",
        )

        if stats:
            assert "total_records" in stats
            assert stats["total_records"] > 0


class TestCacheWarmer:
    """Tests for CacheWarmer."""

    @pytest_asyncio.fixture
    async def mock_redis(self) -> MagicMock:
        """Create mock Redis client."""
        mock = MagicMock(spec=Redis)
        mock.setex = AsyncMock(return_value=True)
        mock.set = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.scan = AsyncMock(return_value=(0, []))
        mock.info = AsyncMock(return_value={
            "keyspace_hits": 100,
            "keyspace_misses": 10,
            "evicted_keys": 0,
            "expired_keys": 5,
        })
        return mock

    @pytest_asyncio.fixture
    async def cache_warmer(self, mock_redis: MagicMock) -> CacheWarmer:
        """Create cache warmer instance."""
        return CacheWarmer(redis_client=mock_redis)

    @pytest.mark.asyncio
    async def test_warmer_initialization(self, cache_warmer: CacheWarmer) -> None:
        """Test cache warmer initialization."""
        assert cache_warmer is not None

    @pytest.mark.asyncio
    async def test_register_strategy(self, cache_warmer: CacheWarmer) -> None:
        """Test registering warming strategy."""
        async def test_strategy() -> list[CacheEntry]:
            return [
                CacheEntry(key="test", value="value", ttl_seconds=60)
            ]

        cache_warmer.register_strategy("test", test_strategy)
        assert "test" in cache_warmer._warming_strategies

    @pytest.mark.asyncio
    async def test_warm_cache(self, cache_warmer: CacheWarmer) -> None:
        """Test cache warming."""
        async def test_strategy() -> list[CacheEntry]:
            return [
                CacheEntry(key="test", value="value", ttl_seconds=60)
            ]

        cache_warmer.register_strategy("test", test_strategy)
        stats = await cache_warmer.warm_cache(strategies=["test"])

        assert isinstance(stats, WarmingStats)
        assert stats.total_keys >= 0

    @pytest.mark.asyncio
    async def test_invalidate_pattern(
        self, cache_warmer: CacheWarmer, mock_redis: MagicMock
    ) -> None:
        """Test cache pattern invalidation."""
        keys_deleted = await cache_warmer.invalidate_pattern("test:*")
        assert keys_deleted >= 0

    @pytest.mark.asyncio
    async def test_get_cache_stats(
        self, cache_warmer: CacheWarmer, mock_redis: MagicMock
    ) -> None:
        """Test getting cache statistics."""
        mock_redis.info.return_value = {
            "keyspace_hits": 100,
            "keyspace_misses": 10,
            "evicted_keys": 0,
            "expired_keys": 5,
        }

        stats = await cache_warmer.get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats


class TestScalingPolicy:
    """Tests for ScalingPolicy."""

    @pytest_asyncio.fixture
    async def scaling_policy(self) -> ScalingPolicy:
        """Create scaling policy instance."""
        return ScalingPolicy(min_instances=1, max_instances=10)

    @pytest.mark.asyncio
    async def test_policy_initialization(self, scaling_policy: ScalingPolicy) -> None:
        """Test scaling policy initialization."""
        assert scaling_policy.current_instances == 1

    @pytest.mark.asyncio
    async def test_add_threshold(self, scaling_policy: ScalingPolicy) -> None:
        """Test adding scaling threshold."""
        threshold = ScalingThreshold(
            metric=MetricType.CPU,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
        )
        scaling_policy.add_threshold(threshold)
        assert MetricType.CPU in scaling_policy.thresholds

    @pytest.mark.asyncio
    async def test_collect_metrics(self, scaling_policy: ScalingPolicy) -> None:
        """Test collecting system metrics."""
        metrics = await scaling_policy.collect_metrics()
        assert isinstance(metrics, ScalingMetrics)
        assert metrics.cpu_percent >= 0

    @pytest.mark.asyncio
    async def test_evaluate_metric(self, scaling_policy: ScalingPolicy) -> None:
        """Test evaluating a metric."""
        threshold = ScalingThreshold(
            metric=MetricType.CPU,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
        )
        scaling_policy.add_threshold(threshold)

        metrics = ScalingMetrics(
            cpu_percent=80.0,
            memory_percent=50.0,
            queue_depth=0,
            request_rate=0.0,
            error_rate=0.0,
            timestamp=datetime.now(UTC),
        )

        direction = scaling_policy.evaluate_metric(MetricType.CPU, metrics)
        assert direction == ScalingDirection.UP

    @pytest.mark.asyncio
    async def test_get_scaling_summary(self, scaling_policy: ScalingPolicy) -> None:
        """Test getting scaling summary."""
        summary = scaling_policy.get_scaling_summary()
        assert "current_instances" in summary
        assert summary["current_instances"] == 1


class TestAdvancedHealthChecker:
    """Tests for AdvancedHealthChecker."""

    @pytest_asyncio.fixture
    async def engine(self) -> AsyncEngine:
        """Create test engine."""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
        )
        yield engine
        await engine.dispose()

    @pytest_asyncio.fixture
    async def mock_redis(self) -> MagicMock:
        """Create mock Redis client."""
        mock = MagicMock(spec=Redis)
        mock.ping = AsyncMock(return_value=True)
        mock.info = AsyncMock(return_value={
            "used_memory_human": "1M",
            "maxmemory_human": "100M",
        })
        return mock

    @pytest_asyncio.fixture
    async def health_checker(
        self, engine: AsyncEngine, mock_redis: MagicMock
    ) -> AdvancedHealthChecker:
        """Create health checker instance."""
        return AdvancedHealthChecker(
            database_engine=engine,
            redis_client=mock_redis,
        )

    @pytest.mark.asyncio
    async def test_checker_initialization(
        self, health_checker: AdvancedHealthChecker
    ) -> None:
        """Test health checker initialization."""
        assert health_checker is not None

    @pytest.mark.asyncio
    async def test_check_database(
        self, health_checker: AdvancedHealthChecker
    ) -> None:
        """Test database health check."""
        health = await health_checker.check_database()
        assert isinstance(health, ComponentHealth)
        assert health.name == "database"
        assert health.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]

    @pytest.mark.asyncio
    async def test_check_redis(self, health_checker: AdvancedHealthChecker) -> None:
        """Test Redis health check."""
        health = await health_checker.check_redis()
        assert isinstance(health, ComponentHealth)
        assert health.name == "redis"
        assert health.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]

    @pytest.mark.asyncio
    async def test_check_connection_pool(
        self, health_checker: AdvancedHealthChecker
    ) -> None:
        """Test connection pool health check."""
        health = await health_checker.check_connection_pool()
        assert isinstance(health, ComponentHealth)
        assert health.name == "connection_pool"

    @pytest.mark.asyncio
    async def test_check_all(self, health_checker: AdvancedHealthChecker) -> None:
        """Test checking all components."""
        system_health = await health_checker.check_all()
        assert system_health.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert len(system_health.components) > 0

    @pytest.mark.asyncio
    async def test_set_latency_thresholds(
        self, health_checker: AdvancedHealthChecker
    ) -> None:
        """Test setting latency thresholds."""
        health_checker.set_latency_thresholds(warning_ms=200, error_ms=1000)
        assert health_checker._latency_warning_ms == 200
        assert health_checker._latency_error_ms == 1000

    @pytest.mark.asyncio
    async def test_set_replication_lag_thresholds(
        self, health_checker: AdvancedHealthChecker
    ) -> None:
        """Test setting replication lag thresholds."""
        health_checker.set_replication_lag_thresholds(
            warning_seconds=5, error_seconds=30
        )
        assert health_checker._replication_lag_warning_seconds == 5
        assert health_checker._replication_lag_error_seconds == 30
