"""Explainability routes for model interpretability and transparency."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.models.user import User
from signalforge.schemas.explainability import (
    AccuracyByFeatureResponse,
    BatchPredictionExplanationRequest,
    ExplanationSummary,
    FeatureAccuracy,
    FeatureImpact,
    FeatureImportance,
    FeatureImportanceHistoryPoint,
    FeatureImportanceHistoryResponse,
    GlobalFeatureImportanceResponse,
    ModelComparisonResponse,
    ModelContribution,
    PredictionExplanationResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/prediction/{prediction_id}", response_model=PredictionExplanationResponse)
async def get_prediction_explanation(
    prediction_id: str,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> PredictionExplanationResponse:
    """Get explanation for a specific prediction.

    Provides detailed feature-level explanation of what drove a specific
    model prediction, including SHAP values and feature impacts.

    Args:
        prediction_id: Unique identifier for the prediction
        db: Database session
        _current_user: Authenticated user

    Returns:
        Detailed prediction explanation with feature impacts

    Raises:
        HTTPException: If prediction not found
    """
    logger.info(
        "Fetching prediction explanation",
        prediction_id=prediction_id,
        user_id=_current_user.id,
    )

    # TODO: Implement actual explainability service integration
    # from signalforge.ml.explainability import ExplainabilityService
    # service = ExplainabilityService(db)
    # explanation = await service.get_prediction_explanation(prediction_id)
    # if not explanation:
    #     raise HTTPException(status_code=404, detail="Prediction not found")

    # Mock implementation
    if prediction_id not in ["pred_001", "pred_002", "pred_003"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction not found: {prediction_id}",
        )

    top_features = [
        FeatureImpact(
            feature_name="RSI_14",
            impact_value=0.42,
            direction="positive",
            description="Strong RSI indicates upward momentum",
        ),
        FeatureImpact(
            feature_name="Volume_Ratio",
            impact_value=0.28,
            direction="positive",
            description="Higher than average volume supports move",
        ),
        FeatureImpact(
            feature_name="MA_Cross_Signal",
            impact_value=-0.15,
            direction="negative",
            description="Moving average crossover suggests caution",
        ),
    ]

    response = PredictionExplanationResponse(
        prediction_id=prediction_id,
        symbol="AAPL",
        timestamp=datetime.now(UTC),
        top_features=top_features,
        explanation_text="Prediction driven by strong RSI momentum and elevated volume, "
        "partially offset by bearish MA crossover signal.",
        confidence_score=0.78,
        shap_base_value=0.5,
    )

    logger.info(
        "Prediction explanation retrieved",
        prediction_id=prediction_id,
        feature_count=len(top_features),
    )

    return response


@router.post("/predictions/batch", response_model=list[PredictionExplanationResponse])
async def get_batch_prediction_explanations(
    request: BatchPredictionExplanationRequest,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> list[PredictionExplanationResponse]:
    """Get explanations for multiple predictions.

    Efficiently retrieves explanations for a batch of predictions,
    useful for bulk analysis and reporting.

    Args:
        request: Batch request with prediction IDs
        db: Database session
        _current_user: Authenticated user

    Returns:
        List of prediction explanations

    Raises:
        HTTPException: If request is invalid
    """
    logger.info(
        "Fetching batch prediction explanations",
        count=len(request.prediction_ids),
        user_id=_current_user.id,
    )

    # TODO: Implement batch explanation service
    # from signalforge.ml.explainability import ExplainabilityService
    # service = ExplainabilityService(db)
    # explanations = await service.get_batch_explanations(request.prediction_ids)

    results: list[PredictionExplanationResponse] = []
    for pred_id in request.prediction_ids:
        try:
            explanation = await get_prediction_explanation(pred_id, _db, _current_user)
            results.append(explanation)
        except HTTPException:
            # Skip predictions that don't exist
            logger.warning("Skipping prediction without explanation", prediction_id=pred_id)
            continue

    logger.info(
        "Batch explanations completed",
        requested=len(request.prediction_ids),
        retrieved=len(results),
    )

    return results


@router.get("/global-importance/{model_id}", response_model=GlobalFeatureImportanceResponse)
async def get_global_feature_importance(
    model_id: str,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    *,
    period_days: int = Query(default=30, ge=1, le=365, description="Analysis period in days"),
) -> GlobalFeatureImportanceResponse:
    """Get global feature importance for a model.

    Calculates aggregate feature importance across all predictions
    made by the model over the specified time period.

    Args:
        model_id: Model identifier
        db: Database session
        _current_user: Authenticated user
        period_days: Number of days to analyze (default 30)

    Returns:
        Global feature importance rankings

    Raises:
        HTTPException: If model not found
    """
    logger.info(
        "Fetching global feature importance",
        model_id=model_id,
        period_days=period_days,
        user_id=_current_user.id,
    )

    # TODO: Implement global importance calculation
    # from signalforge.ml.explainability import ExplainabilityService
    # service = ExplainabilityService(db)
    # importance = await service.calculate_global_importance(model_id, period_days)

    # Mock implementation
    if model_id not in ["ensemble_v1", "lstm_v2"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )

    features = [
        FeatureImportance(feature_name="RSI_14", importance_value=0.18, rank=1),
        FeatureImportance(feature_name="Volume_Ratio", importance_value=0.15, rank=2),
        FeatureImportance(feature_name="MA_Cross_Signal", importance_value=0.12, rank=3),
        FeatureImportance(feature_name="MACD_Histogram", importance_value=0.10, rank=4),
        FeatureImportance(feature_name="Bollinger_Width", importance_value=0.08, rank=5),
    ]

    response = GlobalFeatureImportanceResponse(
        model_id=model_id,
        features=features,
        calculated_at=datetime.now(UTC),
    )

    logger.info(
        "Global feature importance calculated",
        model_id=model_id,
        feature_count=len(features),
    )

    return response


@router.get(
    "/feature-history/{model_id}/{feature_name}",
    response_model=FeatureImportanceHistoryResponse,
)
async def get_feature_importance_history(
    model_id: str,
    feature_name: str,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    *,
    days: int = Query(default=90, ge=1, le=365, description="Historical period in days"),
) -> FeatureImportanceHistoryResponse:
    """Get historical importance of a feature.

    Tracks how a feature's importance has changed over time,
    useful for understanding feature stability and drift.

    Args:
        model_id: Model identifier
        feature_name: Name of the feature
        db: Database session
        _current_user: Authenticated user
        days: Number of days of history (default 90)

    Returns:
        Historical importance values for the feature

    Raises:
        HTTPException: If model or feature not found
    """
    logger.info(
        "Fetching feature importance history",
        model_id=model_id,
        feature_name=feature_name,
        days=days,
        user_id=_current_user.id,
    )

    # TODO: Implement feature importance history query
    # from signalforge.ml.explainability import ExplainabilityService
    # service = ExplainabilityService(db)
    # history = await service.get_feature_history(model_id, feature_name, days)

    # Mock implementation
    if model_id not in ["ensemble_v1", "lstm_v2"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )

    # Generate mock historical data points
    now = datetime.now(UTC)
    history = [
        FeatureImportanceHistoryPoint(
            timestamp=now - timedelta(days=days - i * 10),
            importance_value=0.15 + (i * 0.005),
        )
        for i in range(10)
    ]

    response = FeatureImportanceHistoryResponse(
        model_id=model_id,
        feature_name=feature_name,
        history=history,
    )

    logger.info(
        "Feature importance history retrieved",
        model_id=model_id,
        feature_name=feature_name,
        data_points=len(history),
    )

    return response


@router.get("/model-comparison/{prediction_id}", response_model=ModelComparisonResponse)
async def get_model_comparison(
    prediction_id: str,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> ModelComparisonResponse:
    """Get ensemble model comparison for a prediction.

    Shows how each model in an ensemble contributed to the final
    prediction, including individual predictions and weights.

    Args:
        prediction_id: Prediction identifier
        db: Database session
        _current_user: Authenticated user

    Returns:
        Model comparison with individual contributions

    Raises:
        HTTPException: If prediction not found or not from ensemble
    """
    logger.info(
        "Fetching model comparison",
        prediction_id=prediction_id,
        user_id=_current_user.id,
    )

    # TODO: Implement ensemble comparison service
    # from signalforge.ml.explainability import ExplainabilityService
    # service = ExplainabilityService(db)
    # comparison = await service.get_model_comparison(prediction_id)

    # Mock implementation
    if prediction_id not in ["pred_001", "pred_002", "pred_003"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction not found: {prediction_id}",
        )

    models = [
        ModelContribution(
            model_name="LSTM_Bidirectional",
            prediction=152.5,
            weight=0.4,
            contribution_pct=38.5,
            confidence=0.82,
        ),
        ModelContribution(
            model_name="Transformer_Attention",
            prediction=154.2,
            weight=0.35,
            contribution_pct=35.2,
            confidence=0.79,
        ),
        ModelContribution(
            model_name="GRU_Stacked",
            prediction=151.8,
            weight=0.25,
            contribution_pct=26.3,
            confidence=0.75,
        ),
    ]

    response = ModelComparisonResponse(
        ensemble_id="ensemble_v1",
        prediction_id=prediction_id,
        models=models,
    )

    logger.info(
        "Model comparison retrieved",
        prediction_id=prediction_id,
        model_count=len(models),
    )

    return response


@router.get("/accuracy-by-feature/{model_id}", response_model=AccuracyByFeatureResponse)
async def get_accuracy_by_feature(
    model_id: str,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> AccuracyByFeatureResponse:
    """Get feature-accuracy correlation analysis.

    Analyzes how prediction accuracy varies when different features
    have high vs low values, helping identify feature reliability.

    Args:
        model_id: Model identifier
        db: Database session
        _current_user: Authenticated user

    Returns:
        Feature-accuracy correlation analysis

    Raises:
        HTTPException: If model not found
    """
    logger.info(
        "Fetching accuracy by feature analysis",
        model_id=model_id,
        user_id=_current_user.id,
    )

    # TODO: Implement feature-accuracy analysis
    # from signalforge.ml.explainability import ExplainabilityService
    # service = ExplainabilityService(db)
    # analysis = await service.analyze_accuracy_by_feature(model_id)

    # Mock implementation
    if model_id not in ["ensemble_v1", "lstm_v2"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )

    features = [
        FeatureAccuracy(
            feature_name="RSI_14",
            accuracy_when_high=0.72,
            accuracy_when_low=0.58,
            correlation=0.35,
        ),
        FeatureAccuracy(
            feature_name="Volume_Ratio",
            accuracy_when_high=0.68,
            accuracy_when_low=0.62,
            correlation=0.18,
        ),
        FeatureAccuracy(
            feature_name="MA_Cross_Signal",
            accuracy_when_high=0.65,
            accuracy_when_low=0.66,
            correlation=-0.02,
        ),
    ]

    response = AccuracyByFeatureResponse(
        model_id=model_id,
        features=features,
    )

    logger.info(
        "Accuracy by feature analysis completed",
        model_id=model_id,
        feature_count=len(features),
    )

    return response


@router.get("/summary/{symbol}", response_model=ExplanationSummary)
async def get_explanation_summary(
    symbol: str,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    *,
    model_id: str | None = Query(default=None, description="Specific model to use"),
) -> ExplanationSummary:
    """Get latest explanation summary for a symbol.

    Provides a condensed explanation suitable for dashboard widgets,
    showing the most recent prediction and its key driver.

    Args:
        symbol: Trading symbol
        db: Database session
        _current_user: Authenticated user
        model_id: Optional specific model (defaults to active model)

    Returns:
        Condensed explanation summary

    Raises:
        HTTPException: If symbol has no recent predictions
    """
    symbol = symbol.upper()

    logger.info(
        "Fetching explanation summary",
        symbol=symbol,
        model_id=model_id,
        user_id=_current_user.id,
    )

    # TODO: Implement summary service
    # from signalforge.ml.explainability import ExplainabilityService
    # service = ExplainabilityService(db)
    # summary = await service.get_latest_summary(symbol, model_id)

    # Mock implementation
    if symbol not in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recent predictions found for symbol: {symbol}",
        )

    response = ExplanationSummary(
        symbol=symbol,
        model_id=model_id or "ensemble_v1",
        timestamp=datetime.now(UTC),
        top_feature="RSI_14",
        top_feature_impact=0.42,
        confidence=0.78,
        direction="up",
        explanation="Strong RSI momentum indicates bullish continuation with elevated volume support.",
    )

    logger.info(
        "Explanation summary retrieved",
        symbol=symbol,
        confidence=response.confidence,
    )

    return response
