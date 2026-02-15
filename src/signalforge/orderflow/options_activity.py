"""Options activity detection and analysis."""

from datetime import UTC, datetime, timedelta

import polars as pl
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import FlowDirection, OptionsActivity
from signalforge.orderflow.schemas import (
    OptionsActivityRecord,
    UnusualOptionsActivity,
)

logger = structlog.get_logger(__name__)


class OptionsActivityDetector:
    """Detect and analyze unusual options activity."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize detector.

        Args:
            session: Database session for queries.
        """
        self.session = session
        self.logger = logger.bind(component="options_activity_detector")

    async def detect_unusual_activity(
        self, symbol: str, volume_threshold: float = 2.0
    ) -> list[UnusualOptionsActivity]:
        """Detect unusual options activity.

        Args:
            symbol: Stock symbol.
            volume_threshold: Volume/OI ratio threshold for unusual activity.

        Returns:
            List of unusual options activities.
        """
        start_date = datetime.now(UTC) - timedelta(days=5)

        stmt = select(OptionsActivity).where(
            OptionsActivity.symbol == symbol,
            OptionsActivity.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            self.logger.info("no_options_data", symbol=symbol)
            return []

        df = pl.DataFrame([
            {
                "id": r.id,
                "volume": r.volume,
                "open_interest": r.open_interest,
                "premium": r.premium,
                "option_type": r.option_type,
                "strike": r.strike,
                "expiry": r.expiry,
                "timestamp": r.timestamp,
            }
            for r in records
        ])

        df = df.with_columns([
            (pl.col("volume") / pl.col("open_interest").clip(1)).alias("vol_oi_ratio"),
        ])

        avg_premium = df["premium"].mean() or 0.0
        std_premium = df["premium"].std() or 1.0

        df = df.with_columns([
            ((pl.col("premium") - avg_premium) / std_premium).alias("premium_z_score"),
        ])

        unusual_df = df.filter(
            (pl.col("vol_oi_ratio") >= volume_threshold) |
            (pl.col("premium_z_score").abs() >= 2.0)
        )

        unusual_activities = []
        for row in unusual_df.to_dicts():
            record_obj = next(r for r in records if r.id == row["id"])

            record = OptionsActivityRecord(
                symbol=record_obj.symbol,
                timestamp=record_obj.timestamp,
                option_type=record_obj.option_type,
                strike=record_obj.strike,
                expiry=record_obj.expiry,
                volume=record_obj.volume,
                open_interest=record_obj.open_interest,
                premium=record_obj.premium,
                implied_volatility=record_obj.implied_volatility,
                delta=record_obj.delta,
                is_unusual=True,
            )

            premium_percentile = float(
                (df.filter(pl.col("premium") <= row["premium"]).height / df.height) * 100
            )

            reasons = []
            if row["vol_oi_ratio"] >= volume_threshold:
                reasons.append(f"High volume/OI ratio: {row['vol_oi_ratio']:.2f}")
            if abs(row["premium_z_score"]) >= 2.0:
                reasons.append(f"Unusual premium (z={row['premium_z_score']:.2f})")

            unusual = UnusualOptionsActivity(
                record=record,
                volume_ratio=row["vol_oi_ratio"],
                oi_ratio=row["vol_oi_ratio"],
                premium_percentile=premium_percentile,
                z_score=row["premium_z_score"],
                reason="; ".join(reasons),
            )
            unusual_activities.append(unusual)

        self.logger.info(
            "detected_unusual_activity",
            symbol=symbol,
            total=len(records),
            unusual=len(unusual_activities),
        )

        return unusual_activities

    async def get_put_call_ratio(self, symbol: str) -> float:
        """Calculate put/call volume ratio.

        Args:
            symbol: Stock symbol.

        Returns:
            Put/call ratio.
        """
        start_date = datetime.now(UTC) - timedelta(days=5)

        stmt = select(OptionsActivity).where(
            OptionsActivity.symbol == symbol,
            OptionsActivity.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            return 1.0

        df = pl.DataFrame([
            {"option_type": r.option_type, "volume": r.volume}
            for r in records
        ])

        call_volume = float(
            df.filter(pl.col("option_type") == "call")["volume"].sum()
        )
        put_volume = float(
            df.filter(pl.col("option_type") == "put")["volume"].sum()
        )

        if call_volume == 0:
            return float("inf") if put_volume > 0 else 1.0

        return put_volume / call_volume

    async def get_options_flow_summary(
        self, symbol: str, days: int = 5
    ) -> dict[str, int | float]:
        """Get options flow summary.

        Args:
            symbol: Stock symbol.
            days: Lookback period.

        Returns:
            Summary statistics dictionary.
        """
        start_date = datetime.now(UTC) - timedelta(days=days)

        stmt = select(OptionsActivity).where(
            OptionsActivity.symbol == symbol,
            OptionsActivity.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            return {
                "total_volume": 0,
                "total_premium": 0.0,
                "call_volume": 0,
                "put_volume": 0,
                "put_call_ratio": 1.0,
                "avg_iv": 0.0,
            }

        df = pl.DataFrame([
            {
                "option_type": r.option_type,
                "volume": r.volume,
                "premium": r.premium,
                "implied_volatility": r.implied_volatility or 0.0,
            }
            for r in records
        ])

        call_volume = int(
            df.filter(pl.col("option_type") == "call")["volume"].sum()
        )
        put_volume = int(
            df.filter(pl.col("option_type") == "put")["volume"].sum()
        )

        if call_volume > 0:
            put_call_ratio = put_volume / call_volume
        elif put_volume > 0:
            put_call_ratio = float("inf")
        else:
            put_call_ratio = 1.0

        return {
            "total_volume": int(df["volume"].sum()),
            "total_premium": float(df["premium"].sum()),
            "call_volume": call_volume,
            "put_volume": put_volume,
            "put_call_ratio": put_call_ratio,
            "avg_iv": float(df["implied_volatility"].mean()),  # type: ignore[arg-type]
        }

    async def detect_large_premium_trades(
        self, symbol: str, threshold_usd: float = 100_000
    ) -> list[OptionsActivityRecord]:
        """Detect large premium options trades.

        Args:
            symbol: Stock symbol.
            threshold_usd: Minimum premium threshold.

        Returns:
            List of large premium trades.
        """
        stmt = select(OptionsActivity).where(
            OptionsActivity.symbol == symbol,
            OptionsActivity.premium >= threshold_usd,
        ).order_by(OptionsActivity.timestamp.desc()).limit(100)

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        return [
            OptionsActivityRecord(
                symbol=r.symbol,
                timestamp=r.timestamp,
                option_type=r.option_type,
                strike=r.strike,
                expiry=r.expiry,
                volume=r.volume,
                open_interest=r.open_interest,
                premium=r.premium,
                implied_volatility=r.implied_volatility,
                delta=r.delta,
                is_unusual=True,
            )
            for r in records
        ]

    async def calculate_options_sentiment(self, symbol: str) -> FlowDirection:
        """Calculate options sentiment from flow.

        Args:
            symbol: Stock symbol.

        Returns:
            Flow direction sentiment.
        """
        summary = await self.get_options_flow_summary(symbol, days=5)
        put_call_ratio = summary["put_call_ratio"]

        if isinstance(put_call_ratio, (int, float)):
            if put_call_ratio < 0.7:
                return FlowDirection.BULLISH
            elif put_call_ratio > 1.3:
                return FlowDirection.BEARISH

        return FlowDirection.NEUTRAL

    async def get_expiry_concentration(self, symbol: str) -> dict[str, int]:
        """Get volume concentration by expiry date.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary mapping expiry dates to volumes.
        """
        start_date = datetime.now(UTC) - timedelta(days=30)

        stmt = select(OptionsActivity).where(
            OptionsActivity.symbol == symbol,
            OptionsActivity.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            return {}

        df = pl.DataFrame([
            {"expiry": r.expiry, "volume": r.volume}
            for r in records
        ])

        grouped = df.group_by("expiry").agg(pl.col("volume").sum())

        return {
            str(row["expiry"]): int(row["volume"])
            for row in grouped.to_dicts()
        }
