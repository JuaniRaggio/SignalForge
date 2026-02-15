"""Data retention policies for time-series data management.

This module provides automated data retention policies to manage storage
costs and comply with data retention requirements across different user tiers.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from signalforge.core.logging import LoggerMixin


class UserTier(str, Enum):
    """User subscription tiers with different retention limits."""

    FREE = "free"
    PROSUMER = "prosumer"
    PROFESSIONAL = "professional"


# Retention periods by tier (in days, -1 = unlimited)
RETENTION_BY_TIER: dict[UserTier, int] = {
    UserTier.FREE: 30,
    UserTier.PROSUMER: 365,
    UserTier.PROFESSIONAL: -1,  # unlimited
}


@dataclass
class RetentionConfig:
    """Configuration for a retention policy."""

    table_name: str
    time_column: str
    retention_days: int
    schedule_interval: timedelta = timedelta(days=1)


@dataclass
class RetentionStats:
    """Statistics from a retention policy execution."""

    table_name: str
    rows_deleted: int
    bytes_freed: int | None
    execution_time_seconds: float
    cutoff_date: datetime


class RetentionPolicy(LoggerMixin):
    """Manages data retention policies for time-series tables.

    Features:
    - Tier-based retention policies
    - Automatic data cleanup via TimescaleDB retention policies
    - Manual retention enforcement
    - Retention statistics and reporting
    """

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize retention policy manager.

        Args:
            engine: SQLAlchemy async engine for database operations
        """
        self._engine = engine

    async def add_retention_policy(
        self,
        table_name: str,
        retention_days: int,
        if_not_exists: bool = True,
    ) -> bool:
        """Add automatic retention policy to a hypertable.

        Args:
            table_name: Name of the hypertable
            retention_days: Delete data older than this many days
            if_not_exists: Don't error if policy already exists

        Returns:
            True if policy was added successfully
        """
        if retention_days == -1:
            self.logger.info(
                "unlimited_retention",
                table_name=table_name,
            )
            return True

        try:
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""

            query = f"""
            SELECT add_retention_policy(
                '{table_name}',
                INTERVAL '{retention_days} days',
                {if_not_exists_clause}
            )
            """

            async with self._engine.begin() as conn:
                await conn.execute(text(query))

            self.logger.info(
                "retention_policy_added",
                table_name=table_name,
                retention_days=retention_days,
            )
            return True
        except Exception as e:
            self.logger.error(
                "retention_policy_add_failed",
                table_name=table_name,
                error=str(e),
            )
            return False

    async def remove_retention_policy(
        self,
        table_name: str,
        if_exists: bool = True,
    ) -> bool:
        """Remove retention policy from a hypertable.

        Args:
            table_name: Name of the hypertable
            if_exists: Don't error if policy doesn't exist

        Returns:
            True if policy was removed successfully
        """
        try:
            if_exists_clause = ", IF_EXISTS" if if_exists else ""

            query = f"""
            SELECT remove_retention_policy('{table_name}'{if_exists_clause})
            """

            async with self._engine.begin() as conn:
                await conn.execute(text(query))

            self.logger.info(
                "retention_policy_removed",
                table_name=table_name,
            )
            return True
        except Exception as e:
            self.logger.error(
                "retention_policy_remove_failed",
                table_name=table_name,
                error=str(e),
            )
            return False

    async def apply_retention_policy(
        self,
        tier: UserTier,
        table_name: str,
    ) -> bool:
        """Apply retention policy based on user tier.

        Args:
            tier: User subscription tier
            table_name: Name of the hypertable

        Returns:
            True if policy was applied successfully
        """
        retention_days = RETENTION_BY_TIER[tier]

        if retention_days == -1:
            # Unlimited retention - remove any existing policy
            return await self.remove_retention_policy(table_name)
        else:
            # Add or update retention policy
            return await self.add_retention_policy(table_name, retention_days)

    async def manual_retention_cleanup(
        self,
        table_name: str,
        time_column: str,
        retention_days: int,
    ) -> RetentionStats:
        """Manually clean up old data based on retention period.

        Args:
            table_name: Name of the table
            time_column: Name of the timestamp column
            retention_days: Delete data older than this many days

        Returns:
            Statistics from the cleanup operation
        """
        start_time = datetime.now(UTC)
        cutoff_date = start_time - timedelta(days=retention_days)

        try:
            # Get size before deletion
            size_before_query = f"""
            SELECT pg_total_relation_size('{table_name}')
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(size_before_query))
                size_before = result.scalar()

            # Perform deletion
            delete_query = f"""
            DELETE FROM {table_name}
            WHERE {time_column} < :cutoff_date
            """

            async with self._engine.begin() as conn:
                result = await conn.execute(
                    text(delete_query),
                    {"cutoff_date": cutoff_date},
                )
                rows_deleted = result.rowcount or 0

            # Get size after deletion
            async with self._engine.connect() as conn:
                result = await conn.execute(text(size_before_query))
                size_after = result.scalar()

            bytes_freed = size_before - size_after if size_before and size_after else None
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            stats = RetentionStats(
                table_name=table_name,
                rows_deleted=rows_deleted,
                bytes_freed=bytes_freed,
                execution_time_seconds=execution_time,
                cutoff_date=cutoff_date,
            )

            self.logger.info(
                "manual_retention_cleanup_completed",
                table_name=table_name,
                rows_deleted=rows_deleted,
                bytes_freed=bytes_freed,
                execution_time_seconds=execution_time,
            )

            return stats
        except Exception as e:
            self.logger.error(
                "manual_retention_cleanup_failed",
                table_name=table_name,
                error=str(e),
            )
            return RetentionStats(
                table_name=table_name,
                rows_deleted=0,
                bytes_freed=None,
                execution_time_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                cutoff_date=cutoff_date,
            )

    async def get_data_age_stats(
        self,
        table_name: str,
        time_column: str,
    ) -> dict[str, Any] | None:
        """Get statistics about data age in a table.

        Args:
            table_name: Name of the table
            time_column: Name of the timestamp column

        Returns:
            Dictionary with age statistics
        """
        try:
            query = f"""
            SELECT
                MIN({time_column}) as oldest_record,
                MAX({time_column}) as newest_record,
                COUNT(*) as total_records,
                EXTRACT(EPOCH FROM (MAX({time_column}) - MIN({time_column})))/86400 as age_range_days
            FROM {table_name}
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                row = result.fetchone()

                if row and row[2] > 0:  # total_records > 0
                    return {
                        "table_name": table_name,
                        "oldest_record": row[0],
                        "newest_record": row[1],
                        "total_records": row[2],
                        "age_range_days": float(row[3]) if row[3] else 0.0,
                    }
                return None
        except Exception as e:
            self.logger.error(
                "get_data_age_stats_failed",
                table_name=table_name,
                error=str(e),
            )
            return None

    async def get_retention_policies(self) -> list[dict[str, Any]]:
        """Get all active retention policies.

        Returns:
            List of retention policy dictionaries
        """
        try:
            query = """
            SELECT
                hypertable_name,
                drop_after::text as drop_after
            FROM timescaledb_information.retention_policies
            ORDER BY hypertable_name
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                rows = result.fetchall()

                return [
                    {
                        "table_name": row[0],
                        "drop_after": row[1],
                    }
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(
                "get_retention_policies_failed",
                error=str(e),
            )
            return []

    async def estimate_retention_impact(
        self,
        table_name: str,
        time_column: str,
        retention_days: int,
    ) -> dict[str, Any] | None:
        """Estimate the impact of applying a retention policy.

        Args:
            table_name: Name of the table
            time_column: Name of the timestamp column
            retention_days: Proposed retention period in days

        Returns:
            Dictionary with impact estimates
        """
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)

            query = f"""
            SELECT
                COUNT(*) as total_rows,
                SUM(CASE WHEN {time_column} < :cutoff_date THEN 1 ELSE 0 END) as rows_to_delete,
                pg_total_relation_size('{table_name}') as current_size,
                pg_size_pretty(pg_total_relation_size('{table_name}')) as current_size_pretty
            FROM {table_name}
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(
                    text(query),
                    {"cutoff_date": cutoff_date},
                )
                row = result.fetchone()

                if row:
                    total_rows = row[0]
                    rows_to_delete = row[1]
                    current_size = row[2]

                    # Estimate bytes to free (proportional to rows)
                    if total_rows > 0:
                        estimated_bytes_freed = int(
                            (rows_to_delete / total_rows) * current_size
                        )
                    else:
                        estimated_bytes_freed = 0

                    return {
                        "table_name": table_name,
                        "retention_days": retention_days,
                        "cutoff_date": cutoff_date,
                        "total_rows": total_rows,
                        "rows_to_delete": rows_to_delete,
                        "rows_to_keep": total_rows - rows_to_delete,
                        "deletion_percent": (
                            (rows_to_delete / total_rows * 100) if total_rows > 0 else 0
                        ),
                        "current_size_bytes": current_size,
                        "current_size_pretty": row[3],
                        "estimated_bytes_freed": estimated_bytes_freed,
                    }
                return None
        except Exception as e:
            self.logger.error(
                "estimate_retention_impact_failed",
                table_name=table_name,
                error=str(e),
            )
            return None

    async def configure_table(self, config: RetentionConfig) -> bool:
        """Configure retention for a table with all settings.

        Args:
            config: Retention configuration

        Returns:
            True if configuration was successful
        """
        return await self.add_retention_policy(
            table_name=config.table_name,
            retention_days=config.retention_days,
        )

    def get_tier_retention_days(self, tier: UserTier) -> int:
        """Get retention period for a user tier.

        Args:
            tier: User subscription tier

        Returns:
            Retention period in days (-1 for unlimited)
        """
        return RETENTION_BY_TIER[tier]
