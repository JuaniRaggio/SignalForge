"""Order flow aggregation and analysis."""

from datetime import UTC, datetime, timedelta

import polars as pl
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import FlowDirection, OrderFlowRecord
from signalforge.orderflow.schemas import FlowAggregation

logger = structlog.get_logger(__name__)


class FlowAggregator:
    """Aggregate and analyze order flow across sources."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize aggregator.

        Args:
            session: Database session for queries.
        """
        self.session = session
        self.logger = logger.bind(component="flow_aggregator")

    async def aggregate_flows(
        self, symbol: str, days: int = 5
    ) -> FlowAggregation:
        """Aggregate all flow data for a symbol.

        Args:
            symbol: Stock symbol.
            days: Lookback period in days.

        Returns:
            Aggregated flow analysis.
        """
        start_date = datetime.now(UTC) - timedelta(days=days)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            self.logger.info("no_flow_data", symbol=symbol)
            return FlowAggregation(
                symbol=symbol,
                period_days=days,
                net_flow=0.0,
                bullish_flow=0.0,
                bearish_flow=0.0,
                bias=FlowDirection.NEUTRAL,
                z_score=0.0,
                flow_momentum=0.0,
                dark_pool_volume=0,
                options_premium=0.0,
                short_interest_change=0.0,
            )

        df = pl.DataFrame([
            {
                "value": r.value,
                "direction": r.direction.value,
                "flow_type": r.flow_type.value,
                "volume": r.volume or 0,
                "timestamp": r.timestamp,
            }
            for r in records
        ])

        bullish_flow = float(
            df.filter(pl.col("direction") == "bullish")["value"].sum()
        )
        bearish_flow = float(
            df.filter(pl.col("direction") == "bearish")["value"].sum()
        )
        net_flow = bullish_flow - bearish_flow

        if bullish_flow > bearish_flow * 1.2:
            bias = FlowDirection.BULLISH
        elif bearish_flow > bullish_flow * 1.2:
            bias = FlowDirection.BEARISH
        else:
            bias = FlowDirection.NEUTRAL

        z_score = await self.calculate_flow_z_score(symbol, lookback_days=days * 4)
        flow_momentum = await self.get_flow_momentum(symbol)

        dark_pool_volume = int(
            df.filter(pl.col("flow_type") == "dark_pool")["volume"].sum()
        )
        options_premium = float(
            df.filter(pl.col("flow_type") == "options")["value"].sum()
        )

        return FlowAggregation(
            symbol=symbol,
            period_days=days,
            net_flow=net_flow,
            bullish_flow=bullish_flow,
            bearish_flow=bearish_flow,
            bias=bias,
            z_score=z_score,
            flow_momentum=flow_momentum,
            dark_pool_volume=dark_pool_volume,
            options_premium=options_premium,
            short_interest_change=0.0,
        )

    async def calculate_net_flow(self, symbol: str) -> float:
        """Calculate net flow (bullish - bearish).

        Args:
            symbol: Stock symbol.

        Returns:
            Net flow value.
        """
        aggregation = await self.aggregate_flows(symbol, days=5)
        return aggregation.net_flow

    async def calculate_flow_z_score(
        self, symbol: str, lookback_days: int = 20
    ) -> float:
        """Calculate z-score of current flow vs historical.

        Args:
            symbol: Stock symbol.
            lookback_days: Historical lookback period.

        Returns:
            Z-score of current flow.
        """
        start_date = datetime.now(UTC) - timedelta(days=lookback_days)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.timestamp >= start_date,
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if len(records) < 10:
            return 0.0

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
        ])

        mean_flow = float(daily_net["net_flow"].mean())  # type: ignore[arg-type]
        std_flow = float(daily_net["net_flow"].std())  # type: ignore[arg-type]

        if std_flow == 0:
            return 0.0

        recent_flow = float(daily_net.tail(1)["net_flow"][0])
        z_score = (recent_flow - mean_flow) / std_flow

        return z_score

    async def get_flow_momentum(self, symbol: str) -> float:
        """Calculate momentum of flow direction changes.

        Args:
            symbol: Stock symbol.

        Returns:
            Flow momentum indicator.
        """
        start_date = datetime.now(UTC) - timedelta(days=10)

        stmt = select(OrderFlowRecord).where(
            OrderFlowRecord.symbol == symbol,
            OrderFlowRecord.timestamp >= start_date,
        ).order_by(OrderFlowRecord.timestamp.desc())

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if len(records) < 2:
            return 0.0

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

        if len(daily_net) < 2:
            return 0.0

        recent_avg = float(daily_net.tail(3)["net_flow"].mean())  # type: ignore[arg-type]
        older_avg = float(daily_net.head(len(daily_net) - 3)["net_flow"].mean())  # type: ignore[arg-type]

        if older_avg == 0:
            return 1.0 if recent_avg > 0 else -1.0

        momentum = (recent_avg - older_avg) / abs(older_avg)
        return max(-1.0, min(1.0, momentum))

    async def rank_by_institutional_interest(
        self, symbols: list[str]
    ) -> list[tuple[str, float]]:
        """Rank symbols by institutional interest.

        Args:
            symbols: List of stock symbols to rank.

        Returns:
            List of (symbol, score) tuples sorted by score descending.
        """
        scores: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                aggregation = await self.aggregate_flows(symbol, days=10)

                score = (
                    abs(aggregation.net_flow) / 1_000_000 +
                    abs(aggregation.z_score) * 10 +
                    aggregation.dark_pool_volume / 100_000 +
                    aggregation.options_premium / 1_000_000
                )

                scores.append((symbol, score))
            except Exception as e:
                self.logger.warning(
                    "failed_to_rank_symbol",
                    symbol=symbol,
                    error=str(e),
                )
                scores.append((symbol, 0.0))

        scores.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(
            "ranked_institutional_interest",
            total_symbols=len(symbols),
            top_symbol=scores[0][0] if scores else None,
        )

        return scores
