"""TimescaleDB compression policies for time-series data optimization.

This module provides automated compression policies for TimescaleDB hypertables
to reduce storage requirements and improve query performance for historical data.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from signalforge.core.logging import LoggerMixin


@dataclass
class CompressionConfig:
    """Configuration for a compression policy."""

    table_name: str
    compress_after_days: int
    segment_by_columns: list[str] | None = None
    order_by_columns: list[str] | None = None
    chunk_time_interval: timedelta = timedelta(days=7)


class CompressionPolicy(LoggerMixin):
    """Manages TimescaleDB compression policies for time-series data.

    Compression reduces storage requirements by 90%+ for historical data
    while maintaining full query capabilities with automatic decompression.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize compression policy manager.

        Args:
            engine: SQLAlchemy async engine for database operations
        """
        self._engine = engine

    async def enable_compression(
        self,
        table_name: str,
        segment_by_columns: list[str] | None = None,
        order_by_columns: list[str] | None = None,
    ) -> bool:
        """Enable compression on a TimescaleDB hypertable.

        Args:
            table_name: Name of the hypertable
            segment_by_columns: Columns to segment compressed data by
            order_by_columns: Columns to order compressed data by

        Returns:
            True if compression was enabled successfully
        """
        try:
            # Build ALTER TABLE statement
            alter_parts = [f"ALTER TABLE {table_name} SET (timescaledb.compress = true"]

            if segment_by_columns:
                segment_by = ", ".join(segment_by_columns)
                alter_parts.append(f"timescaledb.compress_segmentby = '{segment_by}'")

            if order_by_columns:
                order_by = ", ".join(f"{col} DESC" for col in order_by_columns)
                alter_parts.append(f"timescaledb.compress_orderby = '{order_by}'")

            alter_statement = ", ".join(alter_parts) + ")"

            async with self._engine.begin() as conn:
                await conn.execute(text(alter_statement))

            self.logger.info(
                "compression_enabled",
                table_name=table_name,
                segment_by=segment_by_columns,
                order_by=order_by_columns,
            )
            return True
        except Exception as e:
            self.logger.error(
                "compression_enable_failed",
                table_name=table_name,
                error=str(e),
            )
            return False

    async def add_compression_policy(
        self,
        table_name: str,
        compress_after_days: int,
        if_not_exists: bool = True,
    ) -> bool:
        """Add automatic compression policy to a hypertable.

        Args:
            table_name: Name of the hypertable
            compress_after_days: Compress chunks older than this many days
            if_not_exists: Don't error if policy already exists

        Returns:
            True if policy was added successfully
        """
        try:
            if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""

            query = f"""
            SELECT add_compression_policy(
                '{table_name}',
                INTERVAL '{compress_after_days} days',
                {if_not_exists_clause}
            )
            """

            async with self._engine.begin() as conn:
                await conn.execute(text(query))

            self.logger.info(
                "compression_policy_added",
                table_name=table_name,
                compress_after_days=compress_after_days,
            )
            return True
        except Exception as e:
            self.logger.error(
                "compression_policy_add_failed",
                table_name=table_name,
                error=str(e),
            )
            return False

    async def remove_compression_policy(
        self,
        table_name: str,
        if_exists: bool = True,
    ) -> bool:
        """Remove compression policy from a hypertable.

        Args:
            table_name: Name of the hypertable
            if_exists: Don't error if policy doesn't exist

        Returns:
            True if policy was removed successfully
        """
        try:
            if_exists_clause = ", IF_EXISTS" if if_exists else ""

            query = f"""
            SELECT remove_compression_policy('{table_name}'{if_exists_clause})
            """

            async with self._engine.begin() as conn:
                await conn.execute(text(query))

            self.logger.info(
                "compression_policy_removed",
                table_name=table_name,
            )
            return True
        except Exception as e:
            self.logger.error(
                "compression_policy_remove_failed",
                table_name=table_name,
                error=str(e),
            )
            return False

    async def compress_chunk(self, chunk_name: str) -> bool:
        """Manually compress a specific chunk.

        Args:
            chunk_name: Name of the chunk to compress

        Returns:
            True if chunk was compressed successfully
        """
        try:
            query = f"SELECT compress_chunk('{chunk_name}')"

            async with self._engine.begin() as conn:
                await conn.execute(text(query))

            self.logger.info(
                "chunk_compressed",
                chunk_name=chunk_name,
            )
            return True
        except Exception as e:
            self.logger.error(
                "chunk_compression_failed",
                chunk_name=chunk_name,
                error=str(e),
            )
            return False

    async def decompress_chunk(self, chunk_name: str) -> bool:
        """Manually decompress a specific chunk.

        Args:
            chunk_name: Name of the chunk to decompress

        Returns:
            True if chunk was decompressed successfully
        """
        try:
            query = f"SELECT decompress_chunk('{chunk_name}')"

            async with self._engine.begin() as conn:
                await conn.execute(text(query))

            self.logger.info(
                "chunk_decompressed",
                chunk_name=chunk_name,
            )
            return True
        except Exception as e:
            self.logger.error(
                "chunk_decompression_failed",
                chunk_name=chunk_name,
                error=str(e),
            )
            return False

    async def get_compression_stats(self, table_name: str) -> dict[str, Any] | None:
        """Get compression statistics for a hypertable.

        Args:
            table_name: Name of the hypertable

        Returns:
            Dictionary with compression statistics
        """
        try:
            query = f"""
            SELECT
                pg_size_pretty(before_compression_total_bytes) as uncompressed_size,
                pg_size_pretty(after_compression_total_bytes) as compressed_size,
                ROUND(
                    100 * (1 - after_compression_total_bytes::numeric /
                    NULLIF(before_compression_total_bytes, 0)),
                    2
                ) as compression_ratio_percent,
                number_compressed_chunks,
                number_uncompressed_chunks
            FROM timescaledb_information.hypertable_compression_stats
            WHERE hypertable_name = '{table_name}'
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                row = result.fetchone()

                if row:
                    return {
                        "table_name": table_name,
                        "uncompressed_size": row[0],
                        "compressed_size": row[1],
                        "compression_ratio_percent": float(row[2]) if row[2] else 0.0,
                        "compressed_chunks": row[3],
                        "uncompressed_chunks": row[4],
                    }
                return None
        except Exception as e:
            self.logger.error(
                "get_compression_stats_failed",
                table_name=table_name,
                error=str(e),
            )
            return None

    async def get_chunk_status(
        self,
        table_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get compression status of chunks for a hypertable.

        Args:
            table_name: Name of the hypertable
            limit: Maximum number of chunks to return

        Returns:
            List of chunk information dictionaries
        """
        try:
            query = f"""
            SELECT
                chunk_name,
                range_start,
                range_end,
                is_compressed,
                pg_size_pretty(total_bytes) as size,
                total_bytes
            FROM timescaledb_information.chunks
            WHERE hypertable_name = '{table_name}'
            ORDER BY range_start DESC
            LIMIT {limit}
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                rows = result.fetchall()

                return [
                    {
                        "chunk_name": row[0],
                        "range_start": row[1],
                        "range_end": row[2],
                        "is_compressed": row[3],
                        "size": row[4],
                        "size_bytes": row[5],
                    }
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(
                "get_chunk_status_failed",
                table_name=table_name,
                error=str(e),
            )
            return []

    async def configure_table(self, config: CompressionConfig) -> bool:
        """Configure compression for a table with all settings.

        Args:
            config: Compression configuration

        Returns:
            True if configuration was successful
        """
        # Enable compression with settings
        success = await self.enable_compression(
            table_name=config.table_name,
            segment_by_columns=config.segment_by_columns,
            order_by_columns=config.order_by_columns,
        )

        if not success:
            return False

        # Add compression policy
        success = await self.add_compression_policy(
            table_name=config.table_name,
            compress_after_days=config.compress_after_days,
        )

        return success

    async def get_all_compression_policies(self) -> list[dict[str, Any]]:
        """Get all active compression policies.

        Returns:
            List of compression policy dictionaries
        """
        try:
            query = """
            SELECT
                hypertable_name,
                compress_after::text as compress_after
            FROM timescaledb_information.compression_settings
            ORDER BY hypertable_name
            """

            async with self._engine.connect() as conn:
                result = await conn.execute(text(query))
                rows = result.fetchall()

                return [
                    {
                        "table_name": row[0],
                        "compress_after": row[1],
                    }
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(
                "get_compression_policies_failed",
                error=str(e),
            )
            return []
