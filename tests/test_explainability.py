"""Comprehensive tests for explainability dashboard widgets."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.explainability.widgets import (
    AccuracyByFeatureWidget,
    ConfidenceIndicatorWidget,
    ExplainabilityDashboardService,
    FeatureImportanceWidget,
    ModelComparisonWidget,
    PredictionExplainerWidget,
)
from signalforge.models.explainability import (
    EnsembleModelComparison,
    FeatureImportanceHistory,
    ImportanceType,
    ModelAccuracyByFeature,
    PredictionExplanation,
)


@pytest.fixture
def mock_redis() -> MagicMock:
    """Create mock Redis client."""
    mock = MagicMock(spec=Redis)
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.scan = AsyncMock(return_value=(0, []))
    return mock


@pytest.fixture
async def sample_prediction_explanation(db_session: AsyncSession) -> PredictionExplanation:
    """Create sample prediction explanation for testing."""
    prediction_id = str(uuid4())
    model_id = "test_model_v1"
    symbol = "AAPL"

    explanation = PredictionExplanation(
        prediction_id=prediction_id,
        model_id=model_id,
        symbol=symbol,
        timestamp=datetime.now(UTC),
        shap_values={
            "volume_ma_10": 0.05,
            "price_momentum": 0.03,
            "rsi": -0.02,
            "macd": 0.01,
            "volatility": -0.015,
        },
        base_value=Decimal("100.0"),
        prediction_value=Decimal("102.5"),
        top_features=[
            {"name": "volume_ma_10", "value": 0.05, "impact": "positive"},
            {"name": "price_momentum", "value": 0.03, "impact": "positive"},
            {"name": "rsi", "value": -0.02, "impact": "negative"},
        ],
        explanation_text="Strong volume and momentum indicators suggest upward movement.",
    )

    db_session.add(explanation)
    await db_session.flush()
    await db_session.refresh(explanation)
    return explanation


@pytest.fixture
async def sample_feature_importance(db_session: AsyncSession) -> list[FeatureImportanceHistory]:
    """Create sample feature importance records."""
    model_id = "test_model_v1"
    calculation_time = datetime.now(UTC)
    period_start = calculation_time - timedelta(days=30)
    period_end = calculation_time

    features = [
        ("volume_ma_10", 0.25),
        ("price_momentum", 0.20),
        ("rsi", 0.15),
        ("macd", 0.12),
        ("volatility", 0.10),
        ("support_level", 0.08),
        ("resistance_level", 0.06),
        ("obv", 0.04),
    ]

    records = []
    for feature_name, importance in features:
        record = FeatureImportanceHistory(
            model_id=model_id,
            feature_name=feature_name,
            importance_value=Decimal(str(importance)),
            importance_type=ImportanceType.SHAP_MEAN,
            calculated_at=calculation_time,
            period_start=period_start,
            period_end=period_end,
        )
        db_session.add(record)
        records.append(record)

    await db_session.flush()
    return records


@pytest.fixture
async def sample_ensemble_comparison(db_session: AsyncSession) -> list[EnsembleModelComparison]:
    """Create sample ensemble model comparison records."""
    ensemble_id = "ensemble_v1"
    prediction_id = str(uuid4())
    timestamp = datetime.now(UTC)

    models = [
        ("lstm_model", 0.4, 105.0, 0.85),
        ("xgboost_model", 0.35, 103.5, 0.82),
        ("random_forest", 0.25, 102.0, 0.78),
    ]

    records = []
    for model_name, weight, prediction, confidence in models:
        record = EnsembleModelComparison(
            ensemble_id=ensemble_id,
            prediction_id=prediction_id,
            model_name=model_name,
            model_weight=Decimal(str(weight)),
            model_prediction=Decimal(str(prediction)),
            model_confidence=Decimal(str(confidence)),
            contribution_to_final=Decimal(str(prediction * weight)),
            timestamp=timestamp,
        )
        db_session.add(record)
        records.append(record)

    await db_session.flush()
    return records


@pytest.fixture
async def sample_accuracy_by_feature(db_session: AsyncSession) -> list[ModelAccuracyByFeature]:
    """Create sample accuracy by feature records."""
    model_id = "test_model_v1"
    calculation_time = datetime.now(UTC)

    features_data = [
        ("volume_ma_10", 0.85, 0.65, 100, 0.75),
        ("price_momentum", 0.82, 0.68, 95, 0.70),
        ("rsi", 0.78, 0.72, 90, 0.45),
    ]

    records = []
    for feature_name, acc_high, acc_low, samples, corr in features_data:
        record = ModelAccuracyByFeature(
            model_id=model_id,
            feature_name=feature_name,
            accuracy_when_high=Decimal(str(acc_high)),
            accuracy_when_low=Decimal(str(acc_low)),
            sample_count=samples,
            correlation_coefficient=Decimal(str(corr)),
            calculated_at=calculation_time,
        )
        db_session.add(record)
        records.append(record)

    await db_session.flush()
    return records


class TestFeatureImportanceWidget:
    """Test suite for FeatureImportanceWidget."""

    async def test_get_data_success(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_feature_importance: list[FeatureImportanceHistory],
    ):
        """Test successful feature importance data retrieval."""
        widget = FeatureImportanceWidget(mock_redis)
        model_id = "test_model_v1"

        result = await widget.get_data(model_id, db_session, top_n=5)

        assert len(result) == 5
        assert result[0]["feature_name"] == "volume_ma_10"
        assert result[0]["importance"] == 0.25
        assert result[0]["percentage"] > 0
        assert result[0]["color_hint"] in ["green", "yellow", "gray"]

        # Verify cache was set
        mock_redis.setex.assert_called_once()

    async def test_get_data_cached(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test cached data retrieval."""
        import json

        cached_data = [
            {
                "feature_name": "test_feature",
                "importance": 0.5,
                "percentage": 50.0,
                "color_hint": "green",
            }
        ]
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data))

        widget = FeatureImportanceWidget(mock_redis)
        result = await widget.get_data("test_model", db_session)

        assert result == cached_data
        mock_redis.get.assert_called_once()
        mock_redis.setex.assert_not_called()

    async def test_get_data_no_features(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling when no features are found."""
        widget = FeatureImportanceWidget(mock_redis)

        result = await widget.get_data("nonexistent_model", db_session)

        assert result == []

    async def test_get_comparison_data(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_feature_importance: list[FeatureImportanceHistory],
    ):
        """Test feature importance comparison across models."""
        widget = FeatureImportanceWidget(mock_redis)

        result = await widget.get_comparison_data(
            ["test_model_v1"], db_session, top_n=3
        )

        assert "models" in result
        assert "features" in result
        assert len(result["models"]) == 1
        assert len(result["features"]) > 0

    async def test_color_hints(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_feature_importance: list[FeatureImportanceHistory],
    ):
        """Test color hint assignment based on importance percentage."""
        widget = FeatureImportanceWidget(mock_redis)
        model_id = "test_model_v1"

        result = await widget.get_data(model_id, db_session, top_n=8)

        # Top features should have green or yellow hints
        assert result[0]["color_hint"] in ["green", "yellow"]

        # Lower importance features should have gray or yellow hints
        if len(result) > 5:
            assert result[-1]["color_hint"] in ["gray", "yellow"]


class TestPredictionExplainerWidget:
    """Test suite for PredictionExplainerWidget."""

    async def test_get_data_success(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test successful prediction explanation retrieval."""
        widget = PredictionExplainerWidget(mock_redis)

        result = await widget.get_data(
            sample_prediction_explanation.prediction_id, db_session
        )

        assert result["base_value"] == 100.0
        assert result["final_value"] == 102.5
        assert len(result["features"]) == 5
        assert "summary_text" in result

        # Verify features have cumulative values
        for feature in result["features"]:
            assert "cumulative" in feature
            assert "impact" in feature
            assert "color_hint" in feature

    async def test_get_data_waterfall_format(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test waterfall chart format with cumulative values."""
        widget = PredictionExplainerWidget(mock_redis)

        result = await widget.get_data(
            sample_prediction_explanation.prediction_id, db_session
        )

        features = result["features"]

        # Verify cumulative calculation
        base = result["base_value"]
        for i, feature in enumerate(features):
            if i == 0:
                expected_cumulative = base + feature["impact"]
            else:
                expected_cumulative = features[i - 1]["cumulative"] + feature["impact"]

            assert abs(feature["cumulative"] - expected_cumulative) < 0.01

    async def test_get_data_no_explanation(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling when no explanation exists."""
        widget = PredictionExplainerWidget(mock_redis)

        result = await widget.get_data(str(uuid4()), db_session)

        assert result["base_value"] == 0.0
        assert result["final_value"] == 0.0
        assert result["features"] == []
        assert "No explanation available" in result["summary_text"]

    async def test_get_latest_for_symbol(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test retrieval of latest explanation for a symbol."""
        widget = PredictionExplainerWidget(mock_redis)

        result = await widget.get_latest_for_symbol("AAPL", db_session)

        assert result["base_value"] == 100.0
        assert result["final_value"] == 102.5
        assert len(result["features"]) > 0

    async def test_get_latest_for_symbol_not_found(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling when symbol has no predictions."""
        widget = PredictionExplainerWidget(mock_redis)

        result = await widget.get_latest_for_symbol("UNKNOWN", db_session)

        assert result["features"] == []
        assert "No predictions available" in result["summary_text"]


class TestModelComparisonWidget:
    """Test suite for ModelComparisonWidget."""

    async def test_get_data_success(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_ensemble_comparison: list[EnsembleModelComparison],
    ):
        """Test successful ensemble comparison data retrieval."""
        widget = ModelComparisonWidget(mock_redis)

        result = await widget.get_data("ensemble_v1", db_session)

        assert "models" in result
        assert "chart_data" in result
        assert len(result["models"]) == 3

        # Verify model data structure
        model = result["models"][0]
        assert "name" in model
        assert "weight_pct" in model
        assert "prediction" in model
        assert "confidence" in model
        assert "contribution" in model

    async def test_get_data_chart_format(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_ensemble_comparison: list[EnsembleModelComparison],
    ):
        """Test chart data formatting."""
        widget = ModelComparisonWidget(mock_redis)

        result = await widget.get_data("ensemble_v1", db_session)

        chart_data = result["chart_data"]
        assert "labels" in chart_data
        assert "weights" in chart_data
        assert "contributions" in chart_data
        assert "confidences" in chart_data

        # Verify data alignment
        assert len(chart_data["labels"]) == len(chart_data["weights"])
        assert len(chart_data["labels"]) == len(chart_data["contributions"])

    async def test_get_data_no_ensemble(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling when ensemble has no data."""
        widget = ModelComparisonWidget(mock_redis)

        result = await widget.get_data("nonexistent_ensemble", db_session)

        assert result["models"] == []
        assert result["chart_data"] == {}

    async def test_get_performance_comparison(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test performance comparison across models."""
        widget = ModelComparisonWidget(mock_redis)

        result = await widget.get_performance_comparison(
            ["test_model_v1"], db_session, days=30
        )

        assert "models" in result
        assert "period_days" in result
        assert result["period_days"] == 30


class TestConfidenceIndicatorWidget:
    """Test suite for ConfidenceIndicatorWidget."""

    async def test_get_data_success(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test successful confidence data retrieval."""
        widget = ConfidenceIndicatorWidget(mock_redis)

        result = await widget.get_data(
            sample_prediction_explanation.prediction_id, db_session
        )

        assert "overall_confidence" in result
        assert "confidence_level" in result
        assert "factors" in result

        assert result["confidence_level"] in ["high", "medium", "low", "unknown"]
        assert 0.0 <= result["overall_confidence"] <= 1.0

    async def test_confidence_level_calculation(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test confidence level is properly calculated."""
        widget = ConfidenceIndicatorWidget(mock_redis)

        result = await widget.get_data(
            sample_prediction_explanation.prediction_id, db_session
        )

        # Should have high confidence due to concentrated SHAP values
        assert result["confidence_level"] in ["high", "medium"]
        assert result["overall_confidence"] > 0.5

    async def test_get_confidence_factors(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test confidence factors extraction."""
        widget = ConfidenceIndicatorWidget(mock_redis)

        factors = await widget.get_confidence_factors(
            sample_prediction_explanation.prediction_id, db_session
        )

        assert len(factors) > 0
        for factor in factors:
            assert "name" in factor
            assert "impact" in factor
            assert "description" in factor
            assert 0.0 <= factor["impact"] <= 1.0

    async def test_get_data_no_prediction(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling when prediction doesn't exist."""
        widget = ConfidenceIndicatorWidget(mock_redis)

        result = await widget.get_data(str(uuid4()), db_session)

        assert result["overall_confidence"] == 0.0
        assert result["confidence_level"] == "unknown"
        assert result["factors"] == []


class TestAccuracyByFeatureWidget:
    """Test suite for AccuracyByFeatureWidget."""

    async def test_get_data_success(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_accuracy_by_feature: list[ModelAccuracyByFeature],
    ):
        """Test successful accuracy by feature data retrieval."""
        widget = AccuracyByFeatureWidget(mock_redis)

        result = await widget.get_data("test_model_v1", db_session)

        assert "features" in result
        assert len(result["features"]) == 3

        # Verify feature data structure
        feature = result["features"][0]
        assert "name" in feature
        assert "accuracy_high" in feature
        assert "accuracy_low" in feature
        assert "correlation" in feature
        assert "sample_count" in feature
        assert "trend" in feature

    async def test_trend_direction(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_accuracy_by_feature: list[ModelAccuracyByFeature],
    ):
        """Test trend direction calculation."""
        widget = AccuracyByFeatureWidget(mock_redis)

        result = await widget.get_data("test_model_v1", db_session)

        for feature in result["features"]:
            if feature["accuracy_high"] > feature["accuracy_low"]:
                assert feature["trend"] == "up"
            else:
                assert feature["trend"] == "down"

    async def test_get_trend_historical(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_accuracy_by_feature: list[ModelAccuracyByFeature],
    ):
        """Test historical trend retrieval."""
        widget = AccuracyByFeatureWidget(mock_redis)

        result = await widget.get_trend(
            "test_model_v1", "volume_ma_10", db_session, days=30
        )

        assert "feature_name" in result
        assert "period_days" in result
        assert "data_points" in result
        assert result["feature_name"] == "volume_ma_10"
        assert result["period_days"] == 30

    async def test_get_data_no_accuracy(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling when no accuracy data exists."""
        widget = AccuracyByFeatureWidget(mock_redis)

        result = await widget.get_data("nonexistent_model", db_session)

        assert result["features"] == []


class TestExplainabilityDashboardService:
    """Test suite for ExplainabilityDashboardService."""

    async def test_get_full_dashboard(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
        sample_feature_importance: list[FeatureImportanceHistory],
        sample_accuracy_by_feature: list[ModelAccuracyByFeature],
    ):
        """Test full dashboard data retrieval."""
        service = ExplainabilityDashboardService(mock_redis)

        result = await service.get_full_dashboard("AAPL", "test_model_v1", db_session)

        assert "symbol" in result
        assert "model_id" in result
        assert "timestamp" in result
        assert "widgets" in result

        widgets = result["widgets"]
        assert "feature_importance" in widgets
        assert "prediction_explanation" in widgets
        assert "accuracy_by_feature" in widgets

    async def test_get_symbol_summary(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test symbol summary for hover/tooltip."""
        service = ExplainabilityDashboardService(mock_redis)

        result = await service.get_symbol_summary("AAPL", db_session)

        assert result["symbol"] == "AAPL"
        assert result["has_prediction"] is True
        assert "timestamp" in result
        assert "prediction_value" in result
        assert "top_features" in result
        assert len(result["top_features"]) <= 3

    async def test_get_symbol_summary_no_data(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test symbol summary when no data exists."""
        service = ExplainabilityDashboardService(mock_redis)

        result = await service.get_symbol_summary("UNKNOWN", db_session)

        assert result["symbol"] == "UNKNOWN"
        assert result["has_prediction"] is False

    async def test_refresh_cache(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test cache refresh functionality."""
        service = ExplainabilityDashboardService(mock_redis)

        result = await service.refresh_cache("test_model_v1", db_session)

        assert "model_id" in result
        assert "keys_cleared" in result
        assert "timestamp" in result
        assert result["model_id"] == "test_model_v1"

        # Verify scan was called
        mock_redis.scan.assert_called()


class TestCachingBehavior:
    """Test suite for caching behavior across widgets."""

    async def test_cache_ttl_real_time_widgets(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_prediction_explanation: PredictionExplanation,
    ):
        """Test real-time widgets use 5-minute TTL."""
        widget = PredictionExplainerWidget(mock_redis)

        await widget.get_data(sample_prediction_explanation.prediction_id, db_session)

        # Verify 5-minute (300 seconds) TTL
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300

    async def test_cache_ttl_historical_widgets(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_feature_importance: list[FeatureImportanceHistory],
    ):
        """Test historical widgets use 1-hour TTL."""
        widget = FeatureImportanceWidget(mock_redis)

        await widget.get_data("test_model_v1", db_session)

        # Verify 1-hour (3600 seconds) TTL
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 3600

    async def test_cache_key_uniqueness(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
        sample_feature_importance: list[FeatureImportanceHistory],
    ):
        """Test cache keys are unique per widget configuration."""
        widget = FeatureImportanceWidget(mock_redis)

        # Request with different top_n values
        await widget.get_data("test_model_v1", db_session, top_n=5)
        key1 = mock_redis.setex.call_args[0][0]

        await widget.get_data("test_model_v1", db_session, top_n=10)
        key2 = mock_redis.setex.call_args[0][0]

        assert key1 != key2


class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_empty_shap_values(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling of prediction with no SHAP values."""
        prediction = PredictionExplanation(
            prediction_id=str(uuid4()),
            model_id="test_model",
            symbol="TEST",
            timestamp=datetime.now(UTC),
            shap_values={},
            base_value=Decimal("100.0"),
            prediction_value=Decimal("100.0"),
            top_features=[],
            explanation_text="No features available",
        )
        db_session.add(prediction)
        await db_session.flush()

        widget = PredictionExplainerWidget(mock_redis)
        result = await widget.get_data(prediction.prediction_id, db_session)

        assert result["features"] == []
        assert result["base_value"] == 100.0

    async def test_zero_total_importance(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling when total importance is zero."""
        model_id = "zero_importance_model"
        calculation_time = datetime.now(UTC)

        # Create features with zero importance
        record = FeatureImportanceHistory(
            model_id=model_id,
            feature_name="feature1",
            importance_value=Decimal("0.0"),
            importance_type=ImportanceType.SHAP_MEAN,
            calculated_at=calculation_time,
            period_start=calculation_time - timedelta(days=1),
            period_end=calculation_time,
        )
        db_session.add(record)
        await db_session.flush()

        widget = FeatureImportanceWidget(mock_redis)
        result = await widget.get_data(model_id, db_session)

        # Should handle gracefully with 0% percentage
        assert len(result) > 0
        assert result[0]["percentage"] == 0.0

    async def test_large_number_of_features(
        self,
        mock_redis: MagicMock,
        db_session: AsyncSession,
    ):
        """Test handling large number of features."""
        model_id = "large_model"
        calculation_time = datetime.now(UTC)

        # Create 100 features
        for i in range(100):
            record = FeatureImportanceHistory(
                model_id=model_id,
                feature_name=f"feature_{i}",
                importance_value=Decimal(str(1.0 / 100)),
                importance_type=ImportanceType.SHAP_MEAN,
                calculated_at=calculation_time,
                period_start=calculation_time - timedelta(days=1),
                period_end=calculation_time,
            )
            db_session.add(record)

        await db_session.flush()

        widget = FeatureImportanceWidget(mock_redis)
        result = await widget.get_data(model_id, db_session, top_n=10)

        # Should limit to top_n
        assert len(result) == 10
