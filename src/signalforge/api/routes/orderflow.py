"""Order flow routes."""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.models.user import User
from signalforge.orderflow.aggregator import FlowAggregator
from signalforge.orderflow.anomaly_detector import FlowAnomalyDetector
from signalforge.orderflow.dark_pools import DarkPoolProcessor
from signalforge.orderflow.options_activity import OptionsActivityDetector
from signalforge.orderflow.schemas import (
    DarkPoolPrint,
    DarkPoolSummary,
    FlowAggregation,
    FlowAnomaly,
    OptionsActivityRecord,
    ShortInterestRecord,
    UnusualOptionsActivity,
)
from signalforge.orderflow.short_interest import ShortInterestTracker

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/dark-pools/{symbol}", response_model=list[DarkPoolPrint])
async def get_dark_pool_prints(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    threshold_usd: float = Query(
        1_000_000,
        ge=0,
        description="Minimum dollar value threshold for large prints",
    ),
) -> list[DarkPoolPrint]:
    """Get dark pool prints for a symbol.

    Returns large dark pool trades that exceed the specified dollar threshold.
    These prints indicate institutional activity in off-exchange venues.
    """
    symbol = symbol.upper()
    processor = DarkPoolProcessor(db)

    try:
        prints = await processor.detect_large_prints(symbol, threshold_usd=threshold_usd)

        logger.info(
            "Retrieved dark pool prints",
            symbol=symbol,
            threshold=threshold_usd,
            count=len(prints),
        )

        return prints
    except Exception as e:
        logger.error("Failed to get dark pool prints", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dark pool prints: {str(e)}",
        )


@router.get("/dark-pools/{symbol}/summary", response_model=DarkPoolSummary)
async def get_dark_pool_summary(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    days: int = Query(30, ge=1, le=365, description="Lookback period in days"),
) -> DarkPoolSummary:
    """Get dark pool activity summary for a symbol.

    Provides aggregated statistics including total volume, trade count,
    dark pool ratio, and institutional bias.
    """
    symbol = symbol.upper()
    processor = DarkPoolProcessor(db)

    try:
        summary = await processor.get_dark_pool_summary(symbol, days=days)

        logger.info(
            "Retrieved dark pool summary",
            symbol=symbol,
            days=days,
            total_volume=summary.total_volume,
            bias=summary.institutional_bias.value,
        )

        return summary
    except Exception as e:
        logger.error("Failed to get dark pool summary", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dark pool summary: {str(e)}",
        )


@router.get("/options/{symbol}", response_model=list[OptionsActivityRecord])
async def get_options_activity(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    threshold_usd: float = Query(
        100_000,
        ge=0,
        description="Minimum premium threshold in USD",
    ),
) -> list[OptionsActivityRecord]:
    """Get large premium options trades for a symbol.

    Returns options trades with premium values exceeding the threshold,
    which may indicate significant institutional positioning.
    """
    symbol = symbol.upper()
    detector = OptionsActivityDetector(db)

    try:
        trades = await detector.detect_large_premium_trades(
            symbol, threshold_usd=threshold_usd
        )

        logger.info(
            "Retrieved options activity",
            symbol=symbol,
            threshold=threshold_usd,
            count=len(trades),
        )

        return trades
    except Exception as e:
        logger.error("Failed to get options activity", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve options activity: {str(e)}",
        )


@router.get("/options/{symbol}/unusual", response_model=list[UnusualOptionsActivity])
async def get_unusual_options_activity(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    volume_threshold: float = Query(
        2.0,
        ge=0,
        description="Volume/OI ratio threshold for unusual activity",
    ),
) -> list[UnusualOptionsActivity]:
    """Get unusual options activity for a symbol.

    Detects unusual options activity based on volume/open interest ratios
    and premium z-scores. High ratios may indicate informed trading.
    """
    symbol = symbol.upper()
    detector = OptionsActivityDetector(db)

    try:
        unusual = await detector.detect_unusual_activity(
            symbol, volume_threshold=volume_threshold
        )

        logger.info(
            "Retrieved unusual options activity",
            symbol=symbol,
            threshold=volume_threshold,
            count=len(unusual),
        )

        return unusual
    except Exception as e:
        logger.error(
            "Failed to get unusual options activity", symbol=symbol, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect unusual options activity: {str(e)}",
        )


# Specific routes MUST come before generic /{symbol} routes
@router.get("/short-interest/squeeze-candidates", response_model=list[str])
async def get_squeeze_candidates(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    min_short_percent: float = Query(
        20.0,
        ge=0,
        le=100,
        description="Minimum short interest percentage",
    ),
    min_days_to_cover: float = Query(
        5.0,
        ge=0,
        description="Minimum days to cover threshold",
    ),
) -> list[str]:
    """Get potential short squeeze candidates.

    Returns symbols with high short interest and high days to cover,
    which may be vulnerable to short squeezes.
    """
    tracker = ShortInterestTracker(db)

    try:
        candidates = await tracker.detect_short_squeeze_candidates(
            min_short_percent=min_short_percent,
            min_days_to_cover=min_days_to_cover,
        )

        logger.info(
            "Retrieved squeeze candidates",
            min_short_pct=min_short_percent,
            min_days_to_cover=min_days_to_cover,
            count=len(candidates),
        )

        return candidates
    except Exception as e:
        logger.error("Failed to get squeeze candidates", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect squeeze candidates: {str(e)}",
        )


# Generic /{symbol} routes MUST come after specific routes
@router.get("/short-interest/{symbol}", response_model=ShortInterestRecord)
async def get_short_interest(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> ShortInterestRecord:
    """Get current short interest data for a symbol.

    Returns the most recent short interest report including short percentage
    of float and days to cover metrics.
    """
    symbol = symbol.upper()
    tracker = ShortInterestTracker(db)

    try:
        short_interest = await tracker.get_current_short_interest(symbol)

        logger.info(
            "Retrieved current short interest",
            symbol=symbol,
            short_percent=short_interest.short_percent,
        )

        return short_interest
    except ValueError as e:
        logger.warning("Short interest not found", symbol=symbol)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to get short interest", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve short interest: {str(e)}",
        )


@router.get("/short-interest/{symbol}/history", response_model=list[ShortInterestRecord])
async def get_short_interest_history(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    reports: int = Query(12, ge=1, le=52, description="Number of historical reports"),
) -> list[ShortInterestRecord]:
    """Get historical short interest data for a symbol.

    Returns a time series of short interest reports to track changes
    in short positioning over time.
    """
    symbol = symbol.upper()
    tracker = ShortInterestTracker(db)

    try:
        history = await tracker.get_short_interest_history(symbol, reports=reports)

        logger.info(
            "Retrieved short interest history",
            symbol=symbol,
            reports=reports,
            count=len(history),
        )

        return history
    except Exception as e:
        logger.error("Failed to get short interest history", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve short interest history: {str(e)}",
        )


@router.get("/aggregation/{symbol}", response_model=FlowAggregation)
async def get_flow_aggregation(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    days: int = Query(5, ge=1, le=90, description="Lookback period in days"),
) -> FlowAggregation:
    """Get aggregated order flow for a symbol.

    Combines all flow sources (dark pools, options, short interest) into
    a unified view with net flow, bias, and momentum indicators.
    """
    symbol = symbol.upper()
    aggregator = FlowAggregator(db)

    try:
        aggregation = await aggregator.aggregate_flows(symbol, days=days)

        logger.info(
            "Retrieved flow aggregation",
            symbol=symbol,
            days=days,
            net_flow=aggregation.net_flow,
            bias=aggregation.bias.value,
        )

        return aggregation
    except Exception as e:
        logger.error("Failed to get flow aggregation", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to aggregate flows: {str(e)}",
        )


@router.get("/anomalies/{symbol}", response_model=list[FlowAnomaly])
async def get_flow_anomalies(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    sensitivity: float = Query(
        2.0,
        ge=0.5,
        le=5.0,
        description="Z-score threshold for anomaly detection",
    ),
) -> list[FlowAnomaly]:
    """Get detected flow anomalies for a symbol.

    Identifies unusual patterns such as volume spikes, options sweeps,
    accumulation, and distribution patterns.
    """
    symbol = symbol.upper()
    detector = FlowAnomalyDetector(db)

    try:
        anomalies = await detector.detect_anomalies(symbol, sensitivity=sensitivity)

        logger.info(
            "Retrieved flow anomalies",
            symbol=symbol,
            sensitivity=sensitivity,
            count=len(anomalies),
        )

        return anomalies
    except Exception as e:
        logger.error("Failed to get flow anomalies", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect anomalies: {str(e)}",
        )


@router.get("/most-shorted", response_model=list[ShortInterestRecord])
async def get_most_shorted_stocks(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    top_n: int = Query(20, ge=1, le=100, description="Number of top stocks to return"),
) -> list[ShortInterestRecord]:
    """Get most heavily shorted stocks.

    Returns stocks with the highest short interest percentages,
    ranked by short interest of float.
    """
    tracker = ShortInterestTracker(db)

    try:
        most_shorted = await tracker.get_most_shorted_stocks(top_n=top_n)

        logger.info(
            "Retrieved most shorted stocks",
            top_n=top_n,
            count=len(most_shorted),
        )

        return most_shorted
    except Exception as e:
        logger.error("Failed to get most shorted stocks", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve most shorted stocks: {str(e)}",
        )
