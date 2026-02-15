"""NLP API routes for document analysis and signal generation."""

import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.core.logging import get_logger
from signalforge.models.news import NewsArticle
from signalforge.models.user import User
from signalforge.nlp.sector_classifier import get_all_sectors
from signalforge.nlp.sentiment import get_sentiment_analyzer
from signalforge.nlp.signals.aggregator import SignalAggregator
from signalforge.nlp.signals.generator import (
    FinancialDocument,
    NLPSignalGenerator,
)
from signalforge.schemas.nlp import (
    AggregatedSignalResponse,
    AggregatedSignalSchema,
    AggregateSignalsRequest,
    AnalystConsensusResponse,
    AnalystConsensusSchema,
    AnalyzeDocumentRequest,
    AnalyzeDocumentResponse,
    NLPSignalsResponse,
    PriceTargetSchema,
    SectorReportResponse,
    SectorSignalSchema,
    SentimentSchema,
)

logger = get_logger(__name__)

router = APIRouter()


@router.post("/analyze-document", response_model=AnalyzeDocumentResponse)
async def analyze_document(
    request: AnalyzeDocumentRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> AnalyzeDocumentResponse:
    """Analyze a financial document and extract signals.

    This endpoint analyzes a financial document (either by ID or raw text)
    and extracts comprehensive NLP signals including sentiment, price targets,
    and analyst consensus.

    Args:
        request: Document analysis request.
        db: Database session.
        _current_user: Authenticated user.

    Returns:
        Comprehensive NLP analysis results.

    Raises:
        HTTPException: If document not found or analysis fails.
    """
    start_time = time.time()

    logger.info(
        "analyze_document_request",
        document_id=request.document_id,
        has_text=request.text is not None,
        user_id=_current_user.id,
    )

    # Validate request
    if request.document_id is None and request.text is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either document_id or text must be provided",
        )

    # Fetch or create document
    if request.document_id is not None:
        # Fetch existing document
        result = await db.execute(
            select(NewsArticle).where(NewsArticle.id == request.document_id)
        )
        article = result.scalar_one_or_none()

        if article is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {request.document_id} not found",
            )

        document = FinancialDocument(
            id=str(article.id),
            title=article.title,
            content=article.content or "",
            symbols=article.symbols,
            published_at=article.published_at or datetime.now(UTC),
            source=article.source,
            metadata=article.metadata_,
        )
    else:
        # Create document from raw text
        document = FinancialDocument(
            id="temp",
            title=request.title or "Untitled",
            content=request.text or "",
            symbols=request.symbols or [],
            published_at=datetime.now(UTC),
            source="api",
            metadata={},
        )

    # Initialize NLP components
    sentiment_analyzer = get_sentiment_analyzer()
    generator = NLPSignalGenerator(sentiment_analyzer)

    # Generate signals
    try:
        signals = await generator.generate_signals(document, db)
    except Exception as e:
        logger.error("signal_generation_failed", error=str(e), document_id=document.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal generation failed: {str(e)}",
        ) from e

    # Convert to response schema
    processing_time_ms = int((time.time() - start_time) * 1000)

    response = AnalyzeDocumentResponse(
        ticker=signals.ticker,
        sentiment=SentimentSchema(
            score=signals.sentiment.score,
            label=signals.sentiment.label,
            delta_vs_baseline=signals.sentiment.delta_vs_baseline,
            is_new_information=signals.sentiment.is_new_information,
        ),
        price_targets=[
            PriceTargetSchema(
                symbol=pt.symbol,
                target_price=Decimal(str(pt.target_price)),
                current_price=Decimal(str(pt.current_price)) if pt.current_price else None,
                upside_percent=Decimal(str(pt.upside_percent)) if pt.upside_percent else None,
                action=pt.action.value,
                rating=pt.rating.value if pt.rating else None,
                analyst=pt.analyst,
                firm=pt.firm,
                date=pt.date,
                confidence=pt.confidence,
                source_text=pt.source_text,
            )
            for pt in signals.price_targets
        ]
        if request.include_price_targets
        else [],
        analyst_consensus=AnalystConsensusSchema(
            rating=signals.analyst_consensus.rating,
            confidence=signals.analyst_consensus.confidence,
            bullish_count=signals.analyst_consensus.bullish_count,
            bearish_count=signals.analyst_consensus.bearish_count,
            neutral_count=signals.analyst_consensus.neutral_count,
        ),
        sector_signals=[
            SectorSignalSchema(
                sector=ss.sector,
                signal=ss.signal,
                confidence=ss.confidence,
                reasoning=ss.reasoning,
            )
            for ss in signals.sector_signals
        ]
        if request.include_sector_signals
        else [],
        urgency=signals.urgency,
        generated_at=signals.generated_at,
        processing_time_ms=processing_time_ms,
    )

    logger.info(
        "document_analyzed",
        ticker=signals.ticker,
        processing_time_ms=processing_time_ms,
        sentiment_score=signals.sentiment.score,
    )

    return response


@router.get("/signals/{symbol}", response_model=NLPSignalsResponse)
async def get_nlp_signals(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    days: int = Query(7, ge=1, le=90, description="Number of days of history"),
) -> NLPSignalsResponse:
    """Get NLP signals for a symbol over a time period.

    Args:
        symbol: Stock ticker symbol.
        days: Number of days of history to retrieve.
        db: Database session.
        _current_user: Authenticated user.

    Returns:
        Historical NLP signals for the symbol.
    """
    logger.info("get_nlp_signals_request", symbol=symbol, days=days, user_id=_current_user.id)

    cutoff = datetime.now() - timedelta(days=days)

    # Fetch documents for symbol
    result = await db.execute(
        select(NewsArticle)
        .where(NewsArticle.symbols.contains([symbol]))
        .where(NewsArticle.published_at >= cutoff)
        .order_by(NewsArticle.published_at.desc())
        .limit(50)
    )

    articles = result.scalars().all()

    if not articles:
        # No articles found, return empty response
        return NLPSignalsResponse(
            symbol=symbol,
            signals=[],
            count=0,
            period_days=days,
            summary={
                "avg_sentiment": 0.0,
                "total_documents": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
            },
        )

    # Analyze each document
    sentiment_analyzer = get_sentiment_analyzer()
    generator = NLPSignalGenerator(sentiment_analyzer)

    signals_list = []
    sentiment_scores = []
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for article in articles[:10]:  # Limit to 10 most recent
        document = FinancialDocument(
            id=str(article.id),
            title=article.title,
            content=article.content or "",
            symbols=article.symbols,
            published_at=article.published_at or datetime.now(UTC),
            source=article.source,
            metadata=article.metadata_,
        )

        try:
            signals = await generator.generate_signals(document, db)

            # Convert to response schema
            signal_response = AnalyzeDocumentResponse(
                ticker=signals.ticker,
                sentiment=SentimentSchema(
                    score=signals.sentiment.score,
                    label=signals.sentiment.label,
                    delta_vs_baseline=signals.sentiment.delta_vs_baseline,
                    is_new_information=signals.sentiment.is_new_information,
                ),
                price_targets=[
                    PriceTargetSchema(
                        symbol=pt.symbol,
                        target_price=Decimal(str(pt.target_price)),
                        current_price=Decimal(str(pt.current_price))
                        if pt.current_price
                        else None,
                        upside_percent=Decimal(str(pt.upside_percent))
                        if pt.upside_percent
                        else None,
                        action=pt.action.value,
                        rating=pt.rating.value if pt.rating else None,
                        analyst=pt.analyst,
                        firm=pt.firm,
                        date=pt.date,
                        confidence=pt.confidence,
                        source_text=pt.source_text,
                    )
                    for pt in signals.price_targets
                ],
                analyst_consensus=AnalystConsensusSchema(
                    rating=signals.analyst_consensus.rating,
                    confidence=signals.analyst_consensus.confidence,
                    bullish_count=signals.analyst_consensus.bullish_count,
                    bearish_count=signals.analyst_consensus.bearish_count,
                    neutral_count=signals.analyst_consensus.neutral_count,
                ),
                sector_signals=[
                    SectorSignalSchema(
                        sector=ss.sector,
                        signal=ss.signal,
                        confidence=ss.confidence,
                        reasoning=ss.reasoning,
                    )
                    for ss in signals.sector_signals
                ],
                urgency=signals.urgency,
                generated_at=signals.generated_at,
                processing_time_ms=0,
            )

            signals_list.append(signal_response)
            sentiment_scores.append(signals.sentiment.score)

            if signals.sentiment.label == "bullish":
                bullish_count += 1
            elif signals.sentiment.label == "bearish":
                bearish_count += 1
            else:
                neutral_count += 1

        except Exception as e:
            logger.warning("signal_generation_failed_for_article", article_id=article.id, error=str(e))
            continue

    # Calculate summary statistics
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

    response = NLPSignalsResponse(
        symbol=symbol,
        signals=signals_list,
        count=len(signals_list),
        period_days=days,
        summary={
            "avg_sentiment": avg_sentiment,
            "total_documents": len(articles),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
        },
    )

    logger.info(
        "nlp_signals_retrieved",
        symbol=symbol,
        count=len(signals_list),
        avg_sentiment=avg_sentiment,
    )

    return response


@router.get("/sector-report/{sector}", response_model=SectorReportResponse)
async def get_sector_report(
    sector: str,
    db: Annotated[AsyncSession, Depends(get_db)],  # noqa: ARG001
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> SectorReportResponse:
    """Get sector intelligence report.

    Args:
        sector: Sector name (must be valid GICS sector).
        db: Database session.
        _current_user: Authenticated user.

    Returns:
        Comprehensive sector analysis report.

    Raises:
        HTTPException: If sector is invalid.
    """
    logger.info("get_sector_report_request", sector=sector, user_id=_current_user.id)

    # Validate sector
    valid_sectors = get_all_sectors()
    if sector not in valid_sectors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sector. Must be one of: {', '.join(valid_sectors)}",
        )

    # For MVP, return a placeholder response
    # In production, this would aggregate signals across all symbols in the sector
    response = SectorReportResponse(
        sector=sector,
        timestamp=datetime.now(),
        overall_sentiment=SentimentSchema(
            score=0.0,
            label="neutral",
            delta_vs_baseline=0.0,
            is_new_information=False,
        ),
        top_symbols=[],
        analyst_consensus=AnalystConsensusSchema(
            rating="hold",
            confidence=0.5,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
        ),
        key_themes=[],
        average_price_target_upside=None,
        document_count=0,
        high_urgency_count=0,
    )

    logger.info("sector_report_generated", sector=sector)

    return response


@router.get("/analyst-consensus/{symbol}", response_model=AnalystConsensusResponse)
async def get_analyst_consensus(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],  # noqa: ARG001
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> AnalystConsensusResponse:
    """Get aggregated analyst consensus for a symbol.

    Args:
        symbol: Stock ticker symbol.
        db: Database session.
        _current_user: Authenticated user.

    Returns:
        Aggregated analyst consensus and price targets.
    """
    logger.info("get_analyst_consensus_request", symbol=symbol, user_id=_current_user.id)

    # For MVP, return a placeholder response
    # In production, this would aggregate all analyst ratings and targets
    response = AnalystConsensusResponse(
        symbol=symbol,
        timestamp=datetime.now(),
        consensus=AnalystConsensusSchema(
            rating="hold",
            confidence=0.5,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
        ),
        price_targets=[],
        mean_target=None,
        median_target=None,
        high_target=None,
        low_target=None,
        current_price=None,
        implied_upside=None,
        num_analysts=0,
    )

    logger.info("analyst_consensus_retrieved", symbol=symbol)

    return response


@router.post("/aggregate-signals", response_model=AggregatedSignalResponse)
async def aggregate_signals(
    request: AggregateSignalsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> AggregatedSignalResponse:
    """Aggregate ML and NLP signals for a symbol.

    This endpoint combines ML predictions with NLP signals to produce
    a unified trading recommendation.

    Args:
        request: Signal aggregation request.
        db: Database session.
        _current_user: Authenticated user.

    Returns:
        Aggregated signal with unified recommendation.
    """
    logger.info(
        "aggregate_signals_request",
        symbol=request.symbol,
        user_id=_current_user.id,
        ml_weight=request.ml_weight,
        nlp_weight=request.nlp_weight,
    )

    # Validate weights sum to 1.0
    if abs(request.ml_weight + request.nlp_weight - 1.0) > 0.001:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ml_weight and nlp_weight must sum to 1.0",
        )

    ml_prediction = None
    nlp_signals = None
    execution_quality = None

    # TODO: Fetch ML prediction if requested
    # TODO: Generate NLP signals if requested
    # TODO: Fetch execution quality if requested

    # For MVP, create placeholder data
    if request.include_nlp_signals:
        # Generate basic NLP signals
        sentiment_analyzer = get_sentiment_analyzer()
        generator = NLPSignalGenerator(sentiment_analyzer)

        # Fetch most recent document for symbol
        result = await db.execute(
            select(NewsArticle)
            .where(NewsArticle.symbols.contains([request.symbol]))
            .order_by(NewsArticle.published_at.desc())
            .limit(1)
        )

        article = result.scalar_one_or_none()

        if article:
            document = FinancialDocument(
                id=str(article.id),
                title=article.title,
                content=article.content or "",
                symbols=article.symbols,
                published_at=article.published_at or datetime.now(UTC),
                source=article.source,
                metadata=article.metadata_,
            )

            nlp_signals = await generator.generate_signals(document, db)

    # Aggregate signals
    aggregator = SignalAggregator(
        ml_weight=request.ml_weight,
        nlp_weight=request.nlp_weight,
    )

    try:
        aggregated = aggregator.aggregate(ml_prediction, nlp_signals, execution_quality)

        # Apply regime conditioning if specified
        if request.market_regime:
            aggregated = aggregator.apply_regime_conditioning(aggregated, request.market_regime)

    except ValueError as e:
        logger.error("signal_aggregation_failed", error=str(e), symbol=request.symbol)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Convert to response schema
    response = AggregatedSignalResponse(
        timestamp=datetime.now(),
        signal=AggregatedSignalSchema(
            symbol=aggregated.symbol,
            direction=aggregated.direction,
            strength=aggregated.strength,
            confidence=aggregated.confidence,
            ml_contribution=aggregated.ml_contribution,
            nlp_contribution=aggregated.nlp_contribution,
            execution_feasibility=aggregated.execution_feasibility,
            recommendation=aggregated.recommendation,
            explanation=aggregated.explanation,
        ),
        ml_prediction=None,  # TODO: Add ML prediction details
        nlp_signals=None,  # TODO: Add NLP signals details
        execution_quality=None,  # TODO: Add execution quality details
        market_regime=request.market_regime,
    )

    logger.info(
        "signals_aggregated",
        symbol=request.symbol,
        direction=aggregated.direction,
        strength=aggregated.strength,
        confidence=aggregated.confidence,
    )

    return response
