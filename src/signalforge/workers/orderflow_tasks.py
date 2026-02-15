"""Celery tasks for OrderFlow module."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

import structlog
from celery import shared_task

from signalforge.core.database import get_session_context
from signalforge.orderflow.aggregator import FlowAggregator
from signalforge.orderflow.anomaly_detector import FlowAnomalyDetector
from signalforge.orderflow.dark_pools import DarkPoolProcessor
from signalforge.orderflow.options_activity import OptionsActivityDetector
from signalforge.orderflow.short_interest import ShortInterestTracker

logger = structlog.get_logger(__name__)

T = TypeVar("T")

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ"]


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async function in sync context for Celery."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(bind=True, name="signalforge.orderflow.tasks.process_dark_pool_data")  # type: ignore[untyped-decorator]
def process_dark_pool_data(
    _self: Any,  # noqa: ARG001
    data: list[dict[str, str | int | float]] | None = None,
) -> dict[str, Any]:
    """Process new dark pool prints using DarkPoolProcessor."""
    data = data or []
    return run_async(_process_dark_pool_data_async(data))


async def _process_dark_pool_data_async(
    data: list[dict[str, str | int | float]],
) -> dict[str, Any]:
    """Async implementation of dark pool data processing."""
    results: dict[str, Any] = {
        "prints_processed": 0,
        "large_prints": 0,
        "success": False,
    }

    if not data:
        logger.info("no_dark_pool_data_provided")
        results["success"] = True
        return results

    async with get_session_context() as session:
        processor = DarkPoolProcessor(session)

        try:
            prints = await processor.process_ats_data(data)
            results["prints_processed"] = len(prints)
            results["large_prints"] = sum(1 for p in prints if p.is_large)

            logger.info(
                "processed_dark_pool_data",
                total_prints=results["prints_processed"],
                large_prints=results["large_prints"],
            )
            results["success"] = True

        except Exception as e:
            results["error"] = str(e)
            logger.error("failed_to_process_dark_pool_data", error=str(e))

    return results


@shared_task(bind=True, name="signalforge.orderflow.tasks.detect_unusual_options")  # type: ignore[untyped-decorator]
def detect_unusual_options(
    _self: Any,  # noqa: ARG001
    symbols: list[str] | None = None,
    volume_threshold: float = 2.0,
) -> dict[str, Any]:
    """Detect unusual options activity using OptionsActivityDetector."""
    symbols = symbols or DEFAULT_SYMBOLS
    return run_async(_detect_unusual_options_async(symbols, volume_threshold))


async def _detect_unusual_options_async(
    symbols: list[str],
    volume_threshold: float,
) -> dict[str, Any]:
    """Async implementation of unusual options detection."""
    results: dict[str, Any] = {
        "symbols_analyzed": 0,
        "unusual_activities": [],
        "total_unusual": 0,
    }

    async with get_session_context() as session:
        detector = OptionsActivityDetector(session)

        for symbol in symbols:
            try:
                unusual_activities = await detector.detect_unusual_activity(
                    symbol=symbol,
                    volume_threshold=volume_threshold,
                )

                if unusual_activities:
                    results["unusual_activities"].append({
                        "symbol": symbol,
                        "count": len(unusual_activities),
                        "activities": [
                            {
                                "option_type": a.record.option_type,
                                "strike": a.record.strike,
                                "expiry": a.record.expiry.isoformat(),
                                "volume": a.record.volume,
                                "premium": a.record.premium,
                                "reason": a.reason,
                                "z_score": a.z_score,
                            }
                            for a in unusual_activities
                        ],
                    })
                    results["total_unusual"] += len(unusual_activities)

                    logger.info(
                        "detected_unusual_options",
                        symbol=symbol,
                        count=len(unusual_activities),
                    )

                results["symbols_analyzed"] += 1

            except Exception as e:
                logger.error(
                    "failed_to_detect_unusual_options",
                    symbol=symbol,
                    error=str(e),
                )

    results["success"] = True
    return results


@shared_task(bind=True, name="signalforge.orderflow.tasks.update_short_interest")  # type: ignore[untyped-decorator]
def update_short_interest(
    _self: Any,  # noqa: ARG001
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Update short interest data using ShortInterestTracker."""
    symbols = symbols or DEFAULT_SYMBOLS
    return run_async(_update_short_interest_async(symbols))


async def _update_short_interest_async(symbols: list[str]) -> dict[str, Any]:
    """Async implementation of short interest update."""
    results: dict[str, Any] = {
        "symbols_updated": 0,
        "squeeze_candidates": [],
        "significant_changes": [],
    }

    async with get_session_context() as session:
        tracker = ShortInterestTracker(session)

        for symbol in symbols:
            try:
                current_si = await tracker.get_current_short_interest(symbol)

                if (
                    current_si.short_percent >= 20.0
                    and current_si.days_to_cover is not None
                    and current_si.days_to_cover >= 5.0
                ):
                    results["squeeze_candidates"].append({
                        "symbol": symbol,
                        "short_percent": current_si.short_percent,
                        "days_to_cover": current_si.days_to_cover,
                    })

                change = await tracker.calculate_short_interest_change(symbol)
                if change.is_significant:
                    results["significant_changes"].append({
                        "symbol": symbol,
                        "change_percent": change.change_percent,
                        "change_shares": change.change_shares,
                        "is_increasing": change.is_increasing,
                    })

                results["symbols_updated"] += 1
                logger.info("updated_short_interest", symbol=symbol)

            except Exception as e:
                logger.error(
                    "failed_to_update_short_interest",
                    symbol=symbol,
                    error=str(e),
                )

    results["success"] = True
    return results


@shared_task(bind=True, name="signalforge.orderflow.tasks.detect_flow_anomalies")  # type: ignore[untyped-decorator]
def detect_flow_anomalies(
    _self: Any,  # noqa: ARG001
    symbols: list[str] | None = None,
    sensitivity: float = 2.0,
) -> dict[str, Any]:
    """Run anomaly detection using FlowAnomalyDetector."""
    symbols = symbols or DEFAULT_SYMBOLS
    return run_async(_detect_flow_anomalies_async(symbols, sensitivity))


async def _detect_flow_anomalies_async(
    symbols: list[str],
    sensitivity: float,
) -> dict[str, Any]:
    """Async implementation of flow anomaly detection."""
    results: dict[str, Any] = {
        "symbols_analyzed": 0,
        "anomalies_detected": [],
        "total_anomalies": 0,
    }

    async with get_session_context() as session:
        detector = FlowAnomalyDetector(session)

        for symbol in symbols:
            try:
                anomalies = await detector.detect_anomalies(
                    symbol=symbol,
                    sensitivity=sensitivity,
                )

                if anomalies:
                    anomaly_score = await detector.get_anomaly_score(symbol)

                    results["anomalies_detected"].append({
                        "symbol": symbol,
                        "count": len(anomalies),
                        "score": anomaly_score,
                        "anomalies": [
                            {
                                "type": a.anomaly_type,
                                "severity": a.severity.value,
                                "description": a.description,
                                "z_score": a.z_score,
                                "timestamp": a.timestamp.isoformat(),
                            }
                            for a in anomalies
                        ],
                    })
                    results["total_anomalies"] += len(anomalies)

                    logger.info(
                        "detected_flow_anomalies",
                        symbol=symbol,
                        count=len(anomalies),
                        score=anomaly_score,
                    )

                results["symbols_analyzed"] += 1

            except Exception as e:
                logger.error(
                    "failed_to_detect_flow_anomalies",
                    symbol=symbol,
                    error=str(e),
                )

    results["success"] = True
    return results


@shared_task(bind=True, name="signalforge.orderflow.tasks.aggregate_daily_flow")  # type: ignore[untyped-decorator]
def aggregate_daily_flow(
    _self: Any,  # noqa: ARG001
    symbols: list[str] | None = None,
    days: int = 5,
) -> dict[str, Any]:
    """Aggregate daily order flow using FlowAggregator."""
    symbols = symbols or DEFAULT_SYMBOLS
    return run_async(_aggregate_daily_flow_async(symbols, days))


async def _aggregate_daily_flow_async(symbols: list[str], days: int) -> dict[str, Any]:
    """Async implementation of daily flow aggregation."""
    results: dict[str, Any] = {
        "symbols_aggregated": 0,
        "aggregations": [],
    }

    async with get_session_context() as session:
        aggregator = FlowAggregator(session)

        for symbol in symbols:
            try:
                aggregation = await aggregator.aggregate_flows(
                    symbol=symbol,
                    days=days,
                )

                results["aggregations"].append({
                    "symbol": symbol,
                    "period_days": aggregation.period_days,
                    "net_flow": aggregation.net_flow,
                    "bullish_flow": aggregation.bullish_flow,
                    "bearish_flow": aggregation.bearish_flow,
                    "bias": aggregation.bias.value,
                    "z_score": aggregation.z_score,
                    "flow_momentum": aggregation.flow_momentum,
                    "dark_pool_volume": aggregation.dark_pool_volume,
                    "options_premium": aggregation.options_premium,
                })

                logger.info(
                    "aggregated_daily_flow",
                    symbol=symbol,
                    net_flow=aggregation.net_flow,
                    bias=aggregation.bias.value,
                )

                results["symbols_aggregated"] += 1

            except Exception as e:
                logger.error(
                    "failed_to_aggregate_flow",
                    symbol=symbol,
                    error=str(e),
                )

    results["success"] = True
    return results
