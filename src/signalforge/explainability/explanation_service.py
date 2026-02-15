"""Explanation service for managing SHAP-based model explanations.

This service provides database-integrated explanation functionality including
storing, retrieving, and analyzing SHAP explanations for model predictions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import polars as pl
import structlog
from redis.asyncio import Redis
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.exceptions import NotFoundError, ValidationError
from signalforge.explainability.explanation_translator import ExplanationTranslator
from signalforge.explainability.shap_calculator import SHAPCalculator
from signalforge.models.explainability import (
    EnsembleModelComparison,
    FeatureImportanceHistory,
    ImportanceType,
    ModelAccuracyByFeature,
    PredictionExplanation,
)

logger = structlog.get_logger(__name__)


class ExplanationService:
    """Service for managing model explanations and feature importance.

    This service integrates SHAP calculation, database storage, caching,
    and translation of explanations into trader-friendly language.
    """

    # Redis cache TTLs
    EXPLANATION_CACHE_TTL = 3600  # 1 hour
    FEATURE_IMPORTANCE_CACHE_TTL = 1800  # 30 minutes

    def __init__(
        self,
        db: AsyncSession,
        redis: Redis | None = None,
    ) -> None:
        """Initialize explanation service.

        Args:
            db: Database session
            redis: Redis client for caching (optional)
        """
        self.db = db
        self.redis = redis
        self.translator = ExplanationTranslator()

        logger.info(
            "explanation_service_initialized",
            has_redis=redis is not None,
        )

    async def generate_prediction_explanation(
        self,
        prediction_id: UUID,
        model: Any,
        input_data: pl.DataFrame,
        model_id: str,
        symbol: str,
        feature_names: list[str],
        timestamp: datetime | None = None,
    ) -> PredictionExplanation:
        """Generate SHAP explanation for a prediction.

        Args:
            prediction_id: UUID of the prediction
            model: Trained model instance
            input_data: Input features (single row DataFrame)
            model_id: Model identifier
            symbol: Stock symbol
            feature_names: List of feature names
            timestamp: Prediction timestamp (defaults to now)

        Returns:
            PredictionExplanation instance

        Raises:
            ValidationError: If input data is invalid
        """
        if len(input_data) != 1:
            msg = f"Expected single row, got {len(input_data)} rows"
            raise ValidationError(msg)

        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Initialize SHAP calculator
        calculator = SHAPCalculator(
            model=model,
            feature_names=feature_names,
        )

        # Calculate SHAP values
        shap_values, base_value = calculator.calculate_single_prediction_shap(input_data)

        # Get prediction value
        prediction_value = float(model.predict(input_data.select(feature_names).to_numpy())[0])

        # Convert SHAP values to dict
        shap_dict = dict(zip(feature_names, [float(v) for v in shap_values], strict=True))

        # Get top features (top 10)
        sorted_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]

        # Build top features list with detailed info
        top_features = []
        for i, (feat_name, shap_val) in enumerate(sorted_features, 1):
            feature_value = float(input_data.select(feat_name).to_numpy()[0, 0])
            direction = "positive" if shap_val > 0 else "negative"

            top_features.append(
                {
                    "feature_name": feat_name,
                    "feature_value": feature_value,
                    "shap_value": shap_val,
                    "direction": direction,
                    "importance_rank": i,
                }
            )

        # Generate human-readable explanation using translator
        explanation_parts = []
        for feat in top_features[:5]:  # Use top 5 for explanation
            impact_text = self.translator.format_feature_impact(
                str(feat["feature_name"]),
                float(feat["shap_value"]),
                str(feat["direction"]),
            )
            explanation_parts.append(impact_text)

        explanation_text = " ".join(explanation_parts)

        # Limit to 2000 characters
        if len(explanation_text) > 2000:
            explanation_text = explanation_text[:1997] + "..."

        # Create database record
        explanation = PredictionExplanation(
            prediction_id=str(prediction_id),
            model_id=model_id,
            symbol=symbol,
            timestamp=timestamp,
            shap_values=shap_dict,
            base_value=Decimal(str(base_value)),
            prediction_value=Decimal(str(prediction_value)),
            top_features=top_features,
            explanation_text=explanation_text,
        )

        self.db.add(explanation)
        await self.db.flush()
        await self.db.refresh(explanation)

        # Cache the explanation
        if self.redis is not None:
            await self._cache_explanation(str(prediction_id), explanation)

        logger.info(
            "generated_prediction_explanation",
            prediction_id=str(prediction_id),
            model_id=model_id,
            symbol=symbol,
            num_top_features=len(top_features),
        )

        return explanation

    async def get_explanation(
        self,
        prediction_id: UUID,
    ) -> PredictionExplanation:
        """Retrieve stored explanation for a prediction.

        Args:
            prediction_id: UUID of the prediction

        Returns:
            PredictionExplanation instance

        Raises:
            NotFoundError: If explanation not found
        """
        # Try cache first
        if self.redis is not None:
            cached = await self._get_cached_explanation(str(prediction_id))
            if cached is not None:
                logger.debug(
                    "explanation_cache_hit",
                    prediction_id=str(prediction_id),
                )
                return cached

        # Query database
        result = await self.db.execute(
            select(PredictionExplanation).where(
                PredictionExplanation.prediction_id == str(prediction_id)
            )
        )
        explanation = result.scalar_one_or_none()

        if explanation is None:
            logger.warning(
                "explanation_not_found",
                prediction_id=str(prediction_id),
            )
            raise NotFoundError(f"Explanation for prediction {prediction_id} not found")

        # Cache for future requests
        if self.redis is not None:
            await self._cache_explanation(str(prediction_id), explanation)

        logger.debug(
            "explanation_retrieved",
            prediction_id=str(prediction_id),
        )

        return explanation

    async def get_explanations_batch(
        self,
        prediction_ids: list[UUID],
    ) -> list[PredictionExplanation]:
        """Batch retrieve explanations.

        Args:
            prediction_ids: List of prediction UUIDs

        Returns:
            List of PredictionExplanation instances
        """
        if not prediction_ids:
            return []

        # Convert to strings
        id_strings = [str(pid) for pid in prediction_ids]

        # Query database
        result = await self.db.execute(
            select(PredictionExplanation)
            .where(PredictionExplanation.prediction_id.in_(id_strings))
            .order_by(desc(PredictionExplanation.timestamp))
        )
        explanations = list(result.scalars().all())

        logger.info(
            "batch_explanations_retrieved",
            requested=len(prediction_ids),
            found=len(explanations),
        )

        return explanations

    async def calculate_global_importance(
        self,
        model_id: str,
        period_days: int = 30,
    ) -> dict[str, float]:
        """Calculate global feature importance for a model.

        Args:
            model_id: Model identifier
            period_days: Number of days to look back

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Calculate time range
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=period_days)

        # Query explanations in period
        result = await self.db.execute(
            select(PredictionExplanation)
            .where(
                and_(
                    PredictionExplanation.model_id == model_id,
                    PredictionExplanation.timestamp >= start_time,
                    PredictionExplanation.timestamp <= end_time,
                )
            )
            .limit(10000)  # Limit to avoid memory issues
        )
        explanations = list(result.scalars().all())

        if not explanations:
            logger.warning(
                "no_explanations_for_importance",
                model_id=model_id,
                period_days=period_days,
            )
            return {}

        # Aggregate SHAP values
        feature_importance: dict[str, list[float]] = {}

        for exp in explanations:
            for feat_name, shap_val in exp.shap_values.items():
                if feat_name not in feature_importance:
                    feature_importance[feat_name] = []
                feature_importance[feat_name].append(abs(float(shap_val)))

        # Calculate mean absolute importance
        global_importance = {
            feat_name: sum(values) / len(values)
            for feat_name, values in feature_importance.items()
        }

        # Store in history table
        history_record = FeatureImportanceHistory(
            model_id=model_id,
            feature_name="_all_features",
            importance_value=Decimal(str(len(global_importance))),
            importance_type=ImportanceType.SHAP_MEAN,
            calculated_at=datetime.now(UTC),
            period_start=start_time,
            period_end=end_time,
        )
        self.db.add(history_record)

        # Store individual feature importance
        for feat_name, importance in global_importance.items():
            feat_record = FeatureImportanceHistory(
                model_id=model_id,
                feature_name=feat_name,
                importance_value=Decimal(str(importance)),
                importance_type=ImportanceType.SHAP_MEAN,
                calculated_at=datetime.now(UTC),
                period_start=start_time,
                period_end=end_time,
            )
            self.db.add(feat_record)

        await self.db.flush()

        logger.info(
            "calculated_global_importance",
            model_id=model_id,
            period_days=period_days,
            num_features=len(global_importance),
            num_explanations=len(explanations),
        )

        return global_importance

    async def get_feature_importance_history(
        self,
        model_id: str,
        feature_name: str,
        days: int = 90,
    ) -> list[dict[str, Any]]:
        """Get historical feature importance for a specific feature.

        Args:
            model_id: Model identifier
            feature_name: Name of the feature
            days: Number of days to look back

        Returns:
            List of historical importance records
        """
        cutoff_time = datetime.now(UTC) - timedelta(days=days)

        result = await self.db.execute(
            select(FeatureImportanceHistory)
            .where(
                and_(
                    FeatureImportanceHistory.model_id == model_id,
                    FeatureImportanceHistory.feature_name == feature_name,
                    FeatureImportanceHistory.calculated_at >= cutoff_time,
                )
            )
            .order_by(desc(FeatureImportanceHistory.calculated_at))
        )
        history = list(result.scalars().all())

        history_data = [
            {
                "calculated_at": record.calculated_at.isoformat(),
                "importance_value": float(record.importance_value),
                "importance_type": record.importance_type.value,
                "period_start": record.period_start.isoformat(),
                "period_end": record.period_end.isoformat(),
            }
            for record in history
        ]

        logger.debug(
            "retrieved_feature_importance_history",
            model_id=model_id,
            feature_name=feature_name,
            days=days,
            num_records=len(history_data),
        )

        return history_data

    async def compare_ensemble_models(
        self,
        prediction_id: UUID,
    ) -> list[EnsembleModelComparison]:
        """Get ensemble model contributions for a prediction.

        Args:
            prediction_id: UUID of the prediction

        Returns:
            List of EnsembleModelComparison instances

        Raises:
            NotFoundError: If no ensemble comparisons found
        """
        result = await self.db.execute(
            select(EnsembleModelComparison).where(
                EnsembleModelComparison.prediction_id == str(prediction_id)
            )
        )
        comparisons = list(result.scalars().all())

        if not comparisons:
            logger.warning(
                "no_ensemble_comparisons_found",
                prediction_id=str(prediction_id),
            )
            raise NotFoundError(
                f"No ensemble comparisons found for prediction {prediction_id}"
            )

        logger.debug(
            "retrieved_ensemble_comparisons",
            prediction_id=str(prediction_id),
            num_models=len(comparisons),
        )

        return comparisons

    async def calculate_accuracy_by_feature(
        self,
        model_id: str,
    ) -> list[ModelAccuracyByFeature]:
        """Calculate feature-accuracy correlation.

        Args:
            model_id: Model identifier

        Returns:
            List of ModelAccuracyByFeature instances
        """
        # This is a placeholder implementation
        # In production, you would analyze actual prediction accuracy
        # versus feature importance to calculate correlations

        result = await self.db.execute(
            select(ModelAccuracyByFeature)
            .where(ModelAccuracyByFeature.model_id == model_id)
            .order_by(desc(ModelAccuracyByFeature.calculated_at))
            .limit(100)
        )
        accuracy_records = list(result.scalars().all())

        logger.debug(
            "retrieved_accuracy_by_feature",
            model_id=model_id,
            num_records=len(accuracy_records),
        )

        return accuracy_records

    async def _cache_explanation(
        self,
        prediction_id: str,
        explanation: PredictionExplanation,
    ) -> None:
        """Cache explanation in Redis.

        Args:
            prediction_id: Prediction ID
            explanation: PredictionExplanation instance
        """
        if self.redis is None:
            return

        cache_key = f"explanation:{prediction_id}"

        # Convert to dict for caching
        cache_data = {
            "id": explanation.id,
            "prediction_id": explanation.prediction_id,
            "model_id": explanation.model_id,
            "symbol": explanation.symbol,
            "timestamp": explanation.timestamp.isoformat(),
            "shap_values": explanation.shap_values,
            "base_value": float(explanation.base_value),
            "prediction_value": float(explanation.prediction_value),
            "top_features": explanation.top_features,
            "explanation_text": explanation.explanation_text,
            "created_at": explanation.created_at.isoformat(),
        }

        await self.redis.setex(
            cache_key,
            self.EXPLANATION_CACHE_TTL,
            json.dumps(cache_data),
        )

        logger.debug(
            "cached_explanation",
            prediction_id=prediction_id,
            ttl=self.EXPLANATION_CACHE_TTL,
        )

    async def _get_cached_explanation(
        self,
        prediction_id: str,
    ) -> PredictionExplanation | None:
        """Get cached explanation from Redis.

        Args:
            prediction_id: Prediction ID

        Returns:
            PredictionExplanation instance or None if not cached
        """
        if self.redis is None:
            return None

        cache_key = f"explanation:{prediction_id}"
        cached_data = await self.redis.get(cache_key)

        if cached_data is None:
            return None

        # Reconstruct from cached data
        _ = json.loads(cached_data)

        # Note: This returns a dict-like object, not a full ORM instance
        # For simplicity, we return None and let the DB query handle it
        # In production, you might want to reconstruct the full object
        return None
