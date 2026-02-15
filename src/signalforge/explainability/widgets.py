"""Dashboard widget services for explainability.

This module provides pre-computed widget data for the frontend dashboard,
enabling real-time visualization of model explainability metrics, feature
importance, prediction explanations, and model comparisons.

All widget data is cached in Redis for optimal performance with configurable TTLs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.models.explainability import (
    EnsembleModelComparison,
    FeatureImportanceHistory,
    ImportanceType,
    ModelAccuracyByFeature,
    PredictionExplanation,
)

logger = get_logger(__name__)


@dataclass
class FeatureImportanceData:
    """Feature importance widget data structure."""

    feature_name: str
    importance: float
    percentage: float
    color_hint: str


@dataclass
class PredictionExplanationData:
    """Prediction explanation widget data structure."""

    base_value: float
    final_value: float
    features: list[dict[str, Any]]
    summary_text: str


@dataclass
class ModelComparisonData:
    """Model comparison widget data structure."""

    models: list[dict[str, Any]]
    chart_data: dict[str, Any]


@dataclass
class ConfidenceIndicatorData:
    """Confidence indicator widget data structure."""

    overall_confidence: float
    confidence_level: str
    factors: list[dict[str, Any]]


@dataclass
class AccuracyByFeatureData:
    """Accuracy by feature widget data structure."""

    features: list[dict[str, Any]]


class FeatureImportanceWidget:
    """Widget for displaying feature importance metrics.

    Provides top N features with importance values formatted for bar chart display,
    with support for cross-model comparisons.
    """

    def __init__(self, redis_client: Redis) -> None:
        """Initialize feature importance widget.

        Args:
            redis_client: Redis client for caching
        """
        self._redis = redis_client

    async def get_data(
        self,
        model_id: str,
        db: AsyncSession,
        top_n: int = 10,
        importance_type: ImportanceType = ImportanceType.SHAP_MEAN,
    ) -> list[dict[str, Any]]:
        """Get top N features with importance values.

        Args:
            model_id: Model identifier
            db: Database session
            top_n: Number of top features to return
            importance_type: Type of importance calculation

        Returns:
            List of feature importance data formatted for bar chart
        """
        cache_key = f"widget:feature_importance:{model_id}:{importance_type.value}:{top_n}"

        # Try cache first
        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        # Fetch latest feature importance from database
        query = (
            select(FeatureImportanceHistory)
            .where(
                FeatureImportanceHistory.model_id == model_id,
                FeatureImportanceHistory.importance_type == importance_type,
            )
            .order_by(desc(FeatureImportanceHistory.calculated_at))
            .limit(1)
        )

        result = await db.execute(query)
        latest_calculation = result.scalar_one_or_none()

        if not latest_calculation:
            logger.warning("no_feature_importance_found", model_id=model_id)
            return []

        # Get all features from that calculation timestamp
        features_query = (
            select(FeatureImportanceHistory)
            .where(
                FeatureImportanceHistory.model_id == model_id,
                FeatureImportanceHistory.importance_type == importance_type,
                FeatureImportanceHistory.calculated_at == latest_calculation.calculated_at,
            )
            .order_by(desc(FeatureImportanceHistory.importance_value))
            .limit(top_n)
        )

        features_result = await db.execute(features_query)
        features = features_result.scalars().all()

        # Calculate total importance for percentage
        total_importance = sum(float(f.importance_value) for f in features)

        # Format data for frontend
        data = []
        for feature in features:
            importance_val = float(feature.importance_value)
            percentage = (importance_val / total_importance * 100) if total_importance > 0 else 0

            # Color hint: green for high importance, yellow for medium, gray for low
            if percentage >= 20:
                color_hint = "green"
            elif percentage >= 10:
                color_hint = "yellow"
            else:
                color_hint = "gray"

            data.append({
                "feature_name": feature.feature_name,
                "importance": round(importance_val, 4),
                "percentage": round(percentage, 2),
                "color_hint": color_hint,
            })

        # Cache for 1 hour
        await self._redis.setex(cache_key, 3600, json.dumps(data))

        logger.info(
            "feature_importance_data_generated",
            model_id=model_id,
            top_n=top_n,
            features_count=len(data),
        )

        return data

    async def get_comparison_data(
        self,
        model_ids: list[str],
        db: AsyncSession,
        top_n: int = 5,
        importance_type: ImportanceType = ImportanceType.SHAP_MEAN,
    ) -> dict[str, Any]:
        """Compare feature importance across multiple models.

        Args:
            model_ids: List of model identifiers
            db: Database session
            top_n: Number of top features per model
            importance_type: Type of importance calculation

        Returns:
            Comparison data with features grouped by model
        """
        cache_key = f"widget:feature_importance_comparison:{':'.join(sorted(model_ids))}:{importance_type.value}:{top_n}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        comparison_data: dict[str, Any] = {"models": [], "features": []}

        for model_id in model_ids:
            model_data = await self.get_data(model_id, db, top_n, importance_type)
            comparison_data["models"].append({
                "model_id": model_id,
                "features": model_data,
            })

        # Extract unique features across all models
        all_features = set()
        for model in comparison_data["models"]:
            for feature in model["features"]:
                all_features.add(feature["feature_name"])

        comparison_data["features"] = sorted(all_features)

        # Cache for 1 hour
        await self._redis.setex(cache_key, 3600, json.dumps(comparison_data))

        logger.info(
            "feature_importance_comparison_generated",
            model_count=len(model_ids),
            unique_features=len(comparison_data["features"]),
        )

        return comparison_data


class PredictionExplainerWidget:
    """Widget for displaying prediction explanations.

    Provides SHAP-based explanations formatted for waterfall chart visualization,
    showing how features contribute to individual predictions.
    """

    def __init__(self, redis_client: Redis) -> None:
        """Initialize prediction explainer widget.

        Args:
            redis_client: Redis client for caching
        """
        self._redis = redis_client

    async def get_data(
        self,
        prediction_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Get explanation data formatted for waterfall chart.

        Args:
            prediction_id: Prediction identifier
            db: Database session

        Returns:
            Explanation data with base value, features, and cumulative impacts
        """
        cache_key = f"widget:prediction_explanation:{prediction_id}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        # Fetch explanation from database
        query = select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id
        )

        result = await db.execute(query)
        explanation = result.scalar_one_or_none()

        if not explanation:
            logger.warning("no_explanation_found", prediction_id=prediction_id)
            return {
                "base_value": 0.0,
                "final_value": 0.0,
                "features": [],
                "summary_text": "No explanation available",
            }

        # Format features for waterfall chart
        features = []
        cumulative = float(explanation.base_value)

        # Sort by absolute SHAP value (impact)
        shap_items = sorted(
            explanation.shap_values.items(),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )

        for feature_name, shap_value in shap_items:
            impact = float(shap_value)
            cumulative += impact

            features.append({
                "name": feature_name,
                "impact": round(impact, 4),
                "cumulative": round(cumulative, 4),
                "color_hint": "green" if impact > 0 else "red",
            })

        data = {
            "base_value": float(explanation.base_value),
            "final_value": float(explanation.prediction_value),
            "features": features,
            "summary_text": explanation.explanation_text,
        }

        # Cache for 5 minutes (real-time widget)
        await self._redis.setex(cache_key, 300, json.dumps(data))

        logger.info(
            "prediction_explanation_generated",
            prediction_id=prediction_id,
            features_count=len(features),
        )

        return data

    async def get_latest_for_symbol(
        self,
        symbol: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Get latest prediction explanation for a symbol.

        Args:
            symbol: Stock symbol
            db: Database session

        Returns:
            Latest explanation data for the symbol
        """
        cache_key = f"widget:latest_explanation:{symbol}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        # Find latest prediction for symbol
        query = (
            select(PredictionExplanation)
            .where(PredictionExplanation.symbol == symbol)
            .order_by(desc(PredictionExplanation.timestamp))
            .limit(1)
        )

        result = await db.execute(query)
        explanation = result.scalar_one_or_none()

        if not explanation:
            logger.warning("no_latest_explanation_found", symbol=symbol)
            return {
                "base_value": 0.0,
                "final_value": 0.0,
                "features": [],
                "summary_text": f"No predictions available for {symbol}",
            }

        # Reuse get_data method
        data = await self.get_data(explanation.prediction_id, db)

        # Cache for 5 minutes
        await self._redis.setex(cache_key, 300, json.dumps(data))

        return data


class ModelComparisonWidget:
    """Widget for comparing ensemble model contributions.

    Displays how individual models within an ensemble contribute to final predictions,
    including weights, confidence scores, and performance metrics.
    """

    def __init__(self, redis_client: Redis) -> None:
        """Initialize model comparison widget.

        Args:
            redis_client: Redis client for caching
        """
        self._redis = redis_client

    async def get_data(
        self,
        ensemble_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Get ensemble model weights and contributions.

        Args:
            ensemble_id: Ensemble identifier
            db: Database session

        Returns:
            Model weights, predictions, and contribution data
        """
        cache_key = f"widget:model_comparison:{ensemble_id}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        # Get latest ensemble comparison
        query = (
            select(EnsembleModelComparison)
            .where(EnsembleModelComparison.ensemble_id == ensemble_id)
            .order_by(desc(EnsembleModelComparison.timestamp))
            .limit(10)  # Get latest prediction's models
        )

        result = await db.execute(query)
        comparisons = result.scalars().all()

        if not comparisons:
            logger.warning("no_ensemble_data_found", ensemble_id=ensemble_id)
            return {"models": [], "chart_data": {}}

        # Aggregate model data
        models = []
        for comp in comparisons:
            models.append({
                "name": comp.model_name,
                "weight_pct": float(comp.model_weight * 100),
                "prediction": float(comp.model_prediction),
                "confidence": float(comp.model_confidence),
                "contribution": float(comp.contribution_to_final),
            })

        # Prepare chart data
        chart_data = {
            "labels": [m["name"] for m in models],
            "weights": [m["weight_pct"] for m in models],
            "contributions": [m["contribution"] for m in models],
            "confidences": [m["confidence"] for m in models],
        }

        data = {
            "models": models,
            "chart_data": chart_data,
        }

        # Cache for 1 hour
        await self._redis.setex(cache_key, 3600, json.dumps(data))

        logger.info(
            "model_comparison_generated",
            ensemble_id=ensemble_id,
            models_count=len(models),
        )

        return data

    async def get_performance_comparison(
        self,
        model_ids: list[str],
        db: AsyncSession,
        days: int = 30,
    ) -> dict[str, Any]:
        """Compare model performance metrics over time.

        Args:
            model_ids: List of model identifiers
            db: Database session
            days: Number of days to analyze

        Returns:
            Performance comparison with accuracy and contribution metrics
        """
        cache_key = f"widget:performance_comparison:{':'.join(sorted(model_ids))}:{days}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        performance_data = []

        for model_id in model_ids:
            # Get recent predictions for this model
            query = (
                select(PredictionExplanation)
                .where(
                    PredictionExplanation.model_id == model_id,
                    PredictionExplanation.timestamp >= cutoff_date,
                )
            )

            result = await db.execute(query)
            predictions = result.scalars().all()

            if not predictions:
                continue

            # Calculate average prediction value as proxy for contribution
            avg_prediction = sum(float(p.prediction_value) for p in predictions) / len(predictions)

            performance_data.append({
                "model_id": model_id,
                "prediction_count": len(predictions),
                "avg_prediction": round(avg_prediction, 4),
                "recent_contribution": round(avg_prediction, 2),
            })

        data = {
            "models": performance_data,
            "period_days": days,
        }

        # Cache for 1 hour
        await self._redis.setex(cache_key, 3600, json.dumps(data))

        logger.info(
            "performance_comparison_generated",
            model_count=len(model_ids),
            days=days,
        )

        return data


class ConfidenceIndicatorWidget:
    """Widget for displaying prediction confidence breakdown.

    Provides confidence levels with contributing factors and detailed
    explanations of what affects confidence scores.
    """

    def __init__(self, redis_client: Redis) -> None:
        """Initialize confidence indicator widget.

        Args:
            redis_client: Redis client for caching
        """
        self._redis = redis_client

    async def get_data(
        self,
        prediction_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Get confidence breakdown for a prediction.

        Args:
            prediction_id: Prediction identifier
            db: Database session

        Returns:
            Confidence level and breakdown
        """
        cache_key = f"widget:confidence:{prediction_id}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        # Get prediction explanation
        query = select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id
        )

        result = await db.execute(query)
        explanation = result.scalar_one_or_none()

        if not explanation:
            logger.warning("no_confidence_data_found", prediction_id=prediction_id)
            return {
                "overall_confidence": 0.0,
                "confidence_level": "unknown",
                "factors": [],
            }

        # Calculate confidence based on SHAP value concentration
        # High concentration of SHAP values in top features = higher confidence
        total_shap = sum(abs(float(v)) for v in explanation.shap_values.values())
        top_features_shap = sum(
            abs(float(v))
            for f, v in sorted(
                explanation.shap_values.items(),
                key=lambda x: abs(float(x[1])),
                reverse=True,
            )[:3]
        )

        concentration = (top_features_shap / total_shap * 100) if total_shap > 0 else 0

        # Determine confidence level
        if concentration >= 70:
            confidence_level = "high"
            overall_confidence = 0.85
        elif concentration >= 50:
            confidence_level = "medium"
            overall_confidence = 0.65
        else:
            confidence_level = "low"
            overall_confidence = 0.45

        # Create factors
        factors = []

        # Feature concentration factor
        factors.append({
            "name": "Feature Concentration",
            "impact": round(concentration / 100, 2),
            "description": f"Top 3 features account for {concentration:.1f}% of impact",
        })

        # Feature diversity factor
        feature_count = len(explanation.shap_values)
        diversity_score = min(feature_count / 20, 1.0)  # Normalize to max 20 features
        factors.append({
            "name": "Feature Diversity",
            "impact": round(diversity_score, 2),
            "description": f"{feature_count} features contributing to prediction",
        })

        # Prediction magnitude factor
        pred_magnitude = abs(float(explanation.prediction_value))
        magnitude_score = min(pred_magnitude / 10, 1.0)  # Normalize
        factors.append({
            "name": "Prediction Strength",
            "impact": round(magnitude_score, 2),
            "description": f"Prediction magnitude: {pred_magnitude:.2f}",
        })

        data = {
            "overall_confidence": round(overall_confidence, 2),
            "confidence_level": confidence_level,
            "factors": factors,
        }

        # Cache for 5 minutes
        await self._redis.setex(cache_key, 300, json.dumps(data))

        logger.info(
            "confidence_data_generated",
            prediction_id=prediction_id,
            confidence_level=confidence_level,
        )

        return data

    async def get_confidence_factors(
        self,
        prediction_id: str,
        db: AsyncSession,
    ) -> list[dict[str, Any]]:
        """Get detailed factors affecting confidence.

        Args:
            prediction_id: Prediction identifier
            db: Database session

        Returns:
            List of confidence factors with impacts
        """
        data = await self.get_data(prediction_id, db)
        return data.get("factors", [])


class AccuracyByFeatureWidget:
    """Widget for analyzing feature-accuracy correlations.

    Shows which features are associated with accurate predictions and
    tracks historical trends in feature reliability.
    """

    def __init__(self, redis_client: Redis) -> None:
        """Initialize accuracy by feature widget.

        Args:
            redis_client: Redis client for caching
        """
        self._redis = redis_client

    async def get_data(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Get feature-accuracy correlation data.

        Args:
            model_id: Model identifier
            db: Database session

        Returns:
            Features with accuracy correlations
        """
        cache_key = f"widget:accuracy_by_feature:{model_id}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        # Get latest accuracy analysis
        query = (
            select(ModelAccuracyByFeature)
            .where(ModelAccuracyByFeature.model_id == model_id)
            .order_by(desc(ModelAccuracyByFeature.calculated_at))
            .limit(10)
        )

        result = await db.execute(query)
        accuracy_records = result.scalars().all()

        if not accuracy_records:
            logger.warning("no_accuracy_data_found", model_id=model_id)
            return {"features": []}

        features = []
        for record in accuracy_records:
            features.append({
                "name": record.feature_name,
                "accuracy_high": float(record.accuracy_when_high),
                "accuracy_low": float(record.accuracy_when_low),
                "correlation": float(record.correlation_coefficient),
                "sample_count": record.sample_count,
                "trend": "up" if record.accuracy_when_high > record.accuracy_when_low else "down",
            })

        data = {"features": features}

        # Cache for 1 hour
        await self._redis.setex(cache_key, 3600, json.dumps(data))

        logger.info(
            "accuracy_by_feature_generated",
            model_id=model_id,
            features_count=len(features),
        )

        return data

    async def get_trend(
        self,
        model_id: str,
        feature_name: str,
        db: AsyncSession,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get historical trend for a feature's accuracy correlation.

        Args:
            model_id: Model identifier
            feature_name: Feature name
            db: Database session
            days: Number of days of history

        Returns:
            Trend data with timestamps and accuracy values
        """
        cache_key = f"widget:accuracy_trend:{model_id}:{feature_name}:{days}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        query = (
            select(ModelAccuracyByFeature)
            .where(
                ModelAccuracyByFeature.model_id == model_id,
                ModelAccuracyByFeature.feature_name == feature_name,
                ModelAccuracyByFeature.calculated_at >= cutoff_date,
            )
            .order_by(ModelAccuracyByFeature.calculated_at)
        )

        result = await db.execute(query)
        records = result.scalars().all()

        data_points = [
            {
                "timestamp": record.calculated_at.isoformat(),
                "accuracy_high": float(record.accuracy_when_high),
                "accuracy_low": float(record.accuracy_when_low),
                "correlation": float(record.correlation_coefficient),
            }
            for record in records
        ]
        trend_data: dict[str, Any] = {
            "feature_name": feature_name,
            "period_days": days,
            "data_points": data_points,
        }

        # Cache for 1 hour
        await self._redis.setex(cache_key, 3600, json.dumps(trend_data))

        logger.info(
            "accuracy_trend_generated",
            model_id=model_id,
            feature_name=feature_name,
            data_points=len(data_points),
        )

        return trend_data


class ExplainabilityDashboardService:
    """Aggregator service for all explainability widgets.

    Provides a unified interface for fetching complete dashboard data
    and managing cache refresh across all widgets.
    """

    def __init__(self, redis_client: Redis) -> None:
        """Initialize dashboard service.

        Args:
            redis_client: Redis client for caching
        """
        self._redis = redis_client
        self.feature_importance = FeatureImportanceWidget(redis_client)
        self.prediction_explainer = PredictionExplainerWidget(redis_client)
        self.model_comparison = ModelComparisonWidget(redis_client)
        self.confidence_indicator = ConfidenceIndicatorWidget(redis_client)
        self.accuracy_by_feature = AccuracyByFeatureWidget(redis_client)

    async def get_full_dashboard(
        self,
        symbol: str,
        model_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Get complete dashboard data for a symbol and model.

        Args:
            symbol: Stock symbol
            model_id: Model identifier
            db: Database session

        Returns:
            Complete dashboard data with all widgets
        """
        cache_key = f"widget:full_dashboard:{symbol}:{model_id}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        logger.info("generating_full_dashboard", symbol=symbol, model_id=model_id)

        # Fetch all widget data in parallel would be ideal, but we'll do sequential for safety
        widgets: dict[str, Any] = {}
        dashboard_data: dict[str, Any] = {
            "symbol": symbol,
            "model_id": model_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "widgets": widgets,
        }

        # Feature importance
        try:
            widgets["feature_importance"] = await self.feature_importance.get_data(
                model_id, db, top_n=10
            )
        except Exception as e:
            logger.error("feature_importance_widget_failed", error=str(e))
            widgets["feature_importance"] = []

        # Latest prediction explanation for symbol
        try:
            widgets["prediction_explanation"] = await self.prediction_explainer.get_latest_for_symbol(
                symbol, db
            )
        except Exception as e:
            logger.error("prediction_explanation_widget_failed", error=str(e))
            widgets["prediction_explanation"] = {}

        # Accuracy by feature
        try:
            widgets["accuracy_by_feature"] = await self.accuracy_by_feature.get_data(
                model_id, db
            )
        except Exception as e:
            logger.error("accuracy_by_feature_widget_failed", error=str(e))
            widgets["accuracy_by_feature"] = {"features": []}

        # Cache for 5 minutes
        await self._redis.setex(cache_key, 300, json.dumps(dashboard_data))

        logger.info(
            "full_dashboard_generated",
            symbol=symbol,
            model_id=model_id,
        )

        return dashboard_data

    async def get_symbol_summary(
        self,
        symbol: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Get quick summary for symbol hover/tooltip.

        Args:
            symbol: Stock symbol
            db: Database session

        Returns:
            Quick summary data
        """
        cache_key = f"widget:symbol_summary:{symbol}"

        cached = await self._redis.get(cache_key)
        if cached:
            logger.debug("cache_hit", cache_key=cache_key)
            return json.loads(cached)

        # Get latest prediction for symbol
        query = (
            select(PredictionExplanation)
            .where(PredictionExplanation.symbol == symbol)
            .order_by(desc(PredictionExplanation.timestamp))
            .limit(1)
        )

        result = await db.execute(query)
        explanation = result.scalar_one_or_none()

        if not explanation:
            return {
                "symbol": symbol,
                "has_prediction": False,
                "message": "No recent predictions",
            }

        # Get top 3 features
        top_features = sorted(
            explanation.shap_values.items(),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )[:3]

        summary = {
            "symbol": symbol,
            "has_prediction": True,
            "timestamp": explanation.timestamp.isoformat(),
            "prediction_value": float(explanation.prediction_value),
            "top_features": [
                {
                    "name": name,
                    "impact": round(float(value), 4),
                }
                for name, value in top_features
            ],
            "summary": explanation.explanation_text[:200],  # Truncate for tooltip
        }

        # Cache for 5 minutes
        await self._redis.setex(cache_key, 300, json.dumps(summary))

        return summary

    async def refresh_cache(
        self,
        model_id: str,
        _db: AsyncSession,
    ) -> dict[str, Any]:
        """Refresh cached widget data for a model.

        Args:
            model_id: Model identifier
            db: Database session

        Returns:
            Refresh status
        """
        logger.info("refreshing_cache", model_id=model_id)

        refreshed_keys = []

        # Clear pattern-based cache keys for this model
        patterns = [
            f"widget:feature_importance:{model_id}:*",
            f"widget:accuracy_by_feature:{model_id}",
            f"widget:accuracy_trend:{model_id}:*",
            f"widget:full_dashboard:*:{model_id}",
        ]

        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                    refreshed_keys.extend(keys)
                if cursor == 0:
                    break

        logger.info(
            "cache_refreshed",
            model_id=model_id,
            keys_cleared=len(refreshed_keys),
        )

        return {
            "model_id": model_id,
            "keys_cleared": len(refreshed_keys),
            "timestamp": datetime.now(UTC).isoformat(),
        }
