"""Flow anomaly detection and pattern recognition."""

from datetime import UTC, datetime, timedelta

import polars as pl
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import OptionsActivity, OrderFlowRecord
from signalforge.orderflow.schemas import AnomalySeverity, FlowAnomaly

logger = structlog.get_logger(__name__)


class FlowAnomalyDetector:
    """Detect anomalies and patterns in order flow."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize detector.

        Args:
            session: Database session for queries.
        """
        self.session = session
        self.logger = logger.bind(component="flow_anomaly_detector")

    async def detect_anomalies(
        self, symbol: str, sensitivity: float = 2.0
    ) -> list[FlowAnomaly]:
        """Detect all types of flow anomalies.

        Args:
            symbol: Stock symbol.
            sensitivity: Z-score threshold for anomaly detection.

        Returns:
            List of detected anomalies.
        """
        anomalies: list[FlowAnomaly] = []

        if await self.detect_volume_spike(symbol, threshold_std=sensitivity):
            anomalies.append(
                FlowAnomaly(
                    symbol=symbol,
                    timestamp=datetime.now(UTC),
                    anomaly_type="volume_spike",
                    severity=AnomalySeverity.HIGH,
                    description="Unusual volume spike detected",
                    z_score=sensitivity,
                )
            )

        sweep_anomalies = await self.detect_options_sweep(symbol)
        anomalies.extend(sweep_anomalies)

        if await self.detect_accumulation_pattern(symbol, days=10):
            anomalies.append(
                FlowAnomaly(
                    symbol=symbol,
                    timestamp=datetime.now(UTC),
                    anomaly_type="accumulation",
                    severity=AnomalySeverity.MEDIUM,
                    description="Accumulation pattern detected over 10 days",
                    z_score=1.5,
                )
            )

        if await self.detect_distribution_pattern(symbol, days=10):
            anomalies.append(
                FlowAnomaly(
                    symbol=symbol,
                    timestamp=datetime.now(UTC),
                    anomaly_type="distribution",
                    severity=AnomalySeverity.MEDIUM,
                    description="Distribution pattern detected over 10 days",
                    z_score=1.5,
                )
            )

        self.logger.info(
            "detected_anomalies",
            symbol=symbol,
            count=len(anomalies),
            sensitivity=sensitivity,
        )

        return anomalies

    async def detect_volume_spike(
        self, symbol: str, threshold_std: float = 3.0
    ) -> bool:
        """Detect unusual volume spikes.

        Args:
            symbol: Stock symbol.
            threshold_std: Standard deviation threshold.

        Returns:
            True if volume spike detected.
        """
        start_date = datetime.now(UTC) - timedelta(days=30)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if len(records) < 10:
            return False

        df = pl.DataFrame([
            {"volume": r.volume or 0, "date": r.timestamp.date()}
            for r in records
        ])

        daily_volume = df.group_by("date").agg(
            pl.col("volume").sum().alias("total_volume")
        )

        mean_volume = float(daily_volume["total_volume"].mean())  # type: ignore[arg-type]
        std_volume = float(daily_volume["total_volume"].std())  # type: ignore[arg-type]

        if std_volume == 0:
            return False

        recent_volume = float(daily_volume.tail(1)["total_volume"][0])
        z_score = (recent_volume - mean_volume) / std_volume

        return z_score > threshold_std

    async def detect_options_sweep(self, symbol: str) -> list[FlowAnomaly]:
        """Detect options sweeps (same strike, multiple exchanges).

        Args:
            symbol: Stock symbol.

        Returns:
            List of detected sweep anomalies.
        """
        start_date = datetime.now(UTC) - timedelta(hours=4)

        stmt = select(OptionsActivity).where(
            OptionsActivity.symbol == symbol,
            OptionsActivity.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if len(records) < 5:
            return []

        df = pl.DataFrame([
            {
                "strike": r.strike,
                "expiry": r.expiry,
                "option_type": r.option_type,
                "volume": r.volume,
                "premium": r.premium,
                "timestamp": r.timestamp,
            }
            for r in records
        ])

        grouped = df.group_by(["strike", "expiry", "option_type"]).agg([
            pl.col("volume").sum().alias("total_volume"),
            pl.col("premium").sum().alias("total_premium"),
            pl.col("timestamp").count().alias("trade_count"),
        ])

        sweeps = grouped.filter(
            (pl.col("trade_count") >= 3) &
            (pl.col("total_premium") >= 50_000)
        )

        anomalies = []
        for sweep in sweeps.to_dicts():
            severity = AnomalySeverity.CRITICAL if sweep["total_premium"] > 500_000 else AnomalySeverity.HIGH

            anomaly = FlowAnomaly(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                anomaly_type="options_sweep",
                severity=severity,
                description=f"Options sweep: {sweep['option_type']} ${sweep['strike']} strike, {sweep['trade_count']} trades",
                z_score=3.0,
                metadata={
                    "strike": float(sweep["strike"]),
                    "option_type": str(sweep["option_type"]),
                    "total_premium": float(sweep["total_premium"]),
                    "trade_count": int(sweep["trade_count"]),
                },
            )
            anomalies.append(anomaly)

        return anomalies

    async def detect_accumulation_pattern(
        self, symbol: str, days: int = 10
    ) -> bool:
        """Detect accumulation pattern (consistent buying).

        Args:
            symbol: Stock symbol.
            days: Period to analyze.

        Returns:
            True if accumulation pattern detected.
        """
        start_date = datetime.now(UTC) - timedelta(days=days)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if len(records) < days * 2:
            return False

        df = pl.DataFrame([
            {
                "value": r.value,
                "direction": r.direction.value,
                "date": r.timestamp.date(),
            }
            for r in records
        ])

        daily_net = df.group_by("date").agg([
            (
                pl.col("value").filter(pl.col("direction") == "bullish").sum() -
                pl.col("value").filter(pl.col("direction") == "bearish").sum()
            ).alias("net_flow")
        ]).sort("date")

        positive_days = int((daily_net["net_flow"] > 0).sum())
        total_days = len(daily_net)

        if total_days == 0:
            return False

        positive_ratio = positive_days / total_days

        tail_mean = float(daily_net.tail(3)["net_flow"].mean())  # type: ignore[arg-type]
        head_mean = float(daily_net.head(3)["net_flow"].mean())  # type: ignore[arg-type]
        increasing_trend = tail_mean > head_mean

        return positive_ratio >= 0.7 and increasing_trend

    async def detect_distribution_pattern(
        self, symbol: str, days: int = 10
    ) -> bool:
        """Detect distribution pattern (consistent selling).

        Args:
            symbol: Stock symbol.
            days: Period to analyze.

        Returns:
            True if distribution pattern detected.
        """
        start_date = datetime.now(UTC) - timedelta(days=days)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if len(records) < days * 2:
            return False

        df = pl.DataFrame([
            {
                "value": r.value,
                "direction": r.direction.value,
                "date": r.timestamp.date(),
            }
            for r in records
        ])

        daily_net = df.group_by("date").agg([
            (
                pl.col("value").filter(pl.col("direction") == "bullish").sum() -
                pl.col("value").filter(pl.col("direction") == "bearish").sum()
            ).alias("net_flow")
        ]).sort("date")

        negative_days = int((daily_net["net_flow"] < 0).sum())
        total_days = len(daily_net)

        if total_days == 0:
            return False

        negative_ratio = negative_days / total_days

        tail_mean = float(daily_net.tail(3)["net_flow"].mean())  # type: ignore[arg-type]
        head_mean = float(daily_net.head(3)["net_flow"].mean())  # type: ignore[arg-type]
        decreasing_trend = tail_mean < head_mean

        return negative_ratio >= 0.7 and decreasing_trend

    async def get_anomaly_score(self, symbol: str) -> float:
        """Calculate composite anomaly score (0-100).

        Args:
            symbol: Stock symbol.

        Returns:
            Anomaly score from 0 (normal) to 100 (highly anomalous).
        """
        anomalies = await self.detect_anomalies(symbol, sensitivity=2.0)

        if not anomalies:
            return 0.0

        severity_weights = {
            AnomalySeverity.LOW: 10.0,
            AnomalySeverity.MEDIUM: 25.0,
            AnomalySeverity.HIGH: 50.0,
            AnomalySeverity.CRITICAL: 100.0,
        }

        total_score = sum(
            severity_weights.get(a.severity, 0.0) for a in anomalies
        )

        score = min(total_score, 100.0)

        self.logger.info(
            "calculated_anomaly_score",
            symbol=symbol,
            score=score,
            anomaly_count=len(anomalies),
        )

        return score
