"""Dark pool activity detection and analysis."""

from datetime import UTC, datetime, timedelta

import polars as pl
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import FlowDirection, FlowType, OrderFlowRecord
from signalforge.orderflow.schemas import DarkPoolPrint, DarkPoolSummary

logger = structlog.get_logger(__name__)


class DarkPoolProcessor:
    """Process and analyze dark pool trading activity."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize processor.

        Args:
            session: Database session for persistence.
        """
        self.session = session
        self.logger = logger.bind(component="dark_pool_processor")

    async def process_ats_data(self, data: list[dict[str, str | int | float]]) -> list[DarkPoolPrint]:
        """Process Alternative Trading System (ATS) data.

        Args:
            data: List of dark pool trade records.

        Returns:
            List of processed dark pool prints.
        """
        if not data:
            self.logger.info("no_ats_data_to_process")
            return []

        df = pl.DataFrame(data)

        required_cols = {"symbol", "timestamp", "shares", "price", "venue"}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            self.logger.error("missing_required_columns", missing=missing)
            raise ValueError(f"Missing required columns: {missing}")

        df = df.with_columns([
            (pl.col("shares") * pl.col("price")).alias("value"),
        ])

        mean_value = float(df["value"].mean() or 0.0)  # type: ignore[arg-type]
        std_value = float(df["value"].std() or 1.0)  # type: ignore[arg-type]

        df = df.with_columns([
            ((pl.col("value") - mean_value) / std_value).alias("z_score"),
            (pl.col("value") > mean_value + 2 * std_value).alias("is_large"),
        ])

        prints = [
            DarkPoolPrint(
                symbol=str(row["symbol"]),
                timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime) else datetime.fromisoformat(str(row["timestamp"])),
                shares=int(row["shares"]),
                price=float(row["price"]),
                value=float(row["value"]),
                venue=str(row["venue"]),
                is_large=bool(row["is_large"]),
                z_score=float(row["z_score"]) if row["z_score"] is not None else None,
            )
            for row in df.to_dicts()
        ]

        self.logger.info(
            "processed_ats_data",
            total_prints=len(prints),
            large_prints=sum(1 for p in prints if p.is_large),
        )

        return prints

    async def get_dark_pool_summary(
        self, symbol: str, days: int = 30
    ) -> DarkPoolSummary:
        """Get dark pool activity summary for a symbol.

        Args:
            symbol: Stock symbol.
            days: Lookback period in days.

        Returns:
            Dark pool summary statistics.
        """
        start_date = datetime.now(UTC) - timedelta(days=days)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.flow_type == FlowType.DARK_POOL,
            OrderFlowRecord.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            self.logger.info("no_dark_pool_data", symbol=symbol, days=days)
            return DarkPoolSummary(
                symbol=symbol,
                period_days=days,
                total_volume=0,
                total_value=0.0,
                trade_count=0,
                avg_trade_size=0.0,
                dark_pool_ratio=0.0,
                largest_print=None,
                institutional_bias=FlowDirection.NEUTRAL,
            )

        df = pl.DataFrame([
            {
                "volume": r.volume or 0,
                "value": r.value,
                "direction": r.direction.value,
                "timestamp": r.timestamp,
                "price": r.price or 0.0,
            }
            for r in records
        ])

        total_volume = int(df["volume"].sum())
        total_value = float(df["value"].sum())
        trade_count = len(df)
        avg_trade_size = total_value / trade_count if trade_count > 0 else 0.0

        bullish_value = float(df.filter(pl.col("direction") == "bullish")["value"].sum())
        bearish_value = float(df.filter(pl.col("direction") == "bearish")["value"].sum())

        if bullish_value > bearish_value * 1.2:
            bias = FlowDirection.BULLISH
        elif bearish_value > bullish_value * 1.2:
            bias = FlowDirection.BEARISH
        else:
            bias = FlowDirection.NEUTRAL

        largest_idx = df["value"].arg_max()
        largest_record = records[largest_idx] if largest_idx is not None else None

        largest_print = None
        if largest_record:
            largest_print = DarkPoolPrint(
                symbol=largest_record.symbol,
                timestamp=largest_record.timestamp,
                shares=largest_record.volume or 0,
                price=largest_record.price or 0.0,
                value=largest_record.value,
                venue=largest_record.source,
                is_large=largest_record.is_unusual,
                z_score=largest_record.z_score,
            )

        dark_pool_ratio = await self.calculate_dark_pool_ratio(symbol)

        return DarkPoolSummary(
            symbol=symbol,
            period_days=days,
            total_volume=total_volume,
            total_value=total_value,
            trade_count=trade_count,
            avg_trade_size=avg_trade_size,
            dark_pool_ratio=dark_pool_ratio,
            largest_print=largest_print,
            institutional_bias=bias,
        )

    async def detect_large_prints(
        self, symbol: str, threshold_usd: float = 1_000_000
    ) -> list[DarkPoolPrint]:
        """Detect large dark pool prints.

        Args:
            symbol: Stock symbol.
            threshold_usd: Minimum dollar value threshold.

        Returns:
            List of large dark pool prints.
        """
        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.flow_type == FlowType.DARK_POOL,
            OrderFlowRecord.value >= threshold_usd,
        ).order_by(OrderFlowRecord.timestamp.desc()).limit(100)

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        prints = [
            DarkPoolPrint(
                symbol=r.symbol,
                timestamp=r.timestamp,
                shares=r.volume or 0,
                price=r.price or 0.0,
                value=r.value,
                venue=r.source,
                is_large=True,
                z_score=r.z_score,
            )
            for r in records
        ]

        self.logger.info(
            "detected_large_prints",
            symbol=symbol,
            threshold=threshold_usd,
            count=len(prints),
        )

        return prints

    async def calculate_dark_pool_ratio(self, symbol: str) -> float:
        """Calculate percentage of volume traded in dark pools.

        Args:
            symbol: Stock symbol.

        Returns:
            Dark pool ratio (0-1).
        """
        start_date = datetime.now(UTC) - timedelta(days=30)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.flow_type == FlowType.DARK_POOL,
            OrderFlowRecord.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            return 0.0

        dark_volume = sum(r.volume or 0 for r in records)

        total_volume_stmt = select(OrderFlowRecord.volume).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.timestamp >= start_date,
        )

        total_result = await self.session.execute(total_volume_stmt)
        total_volume = sum(v or 0 for v in total_result.scalars().all())

        if total_volume == 0:
            return 0.0

        ratio = dark_volume / total_volume
        return min(ratio, 1.0)

    async def get_institutional_bias(self, symbol: str) -> FlowDirection:
        """Determine institutional bias from dark pool activity.

        Args:
            symbol: Stock symbol.

        Returns:
            Flow direction indicating bias.
        """
        summary = await self.get_dark_pool_summary(symbol, days=10)
        return summary.institutional_bias
