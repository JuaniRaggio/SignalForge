"""Tests for SHAP-based explanation system."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from signalforge.explanations import (
    ExplanationConfig,
    ExplanationVisualizer,
    FeatureContribution,
    NarrativeGenerator,
    PredictionExplanation,
    SHAPExplainer,
)


# Fixtures
@pytest.fixture
def mock_model() -> Mock:
    """Create a mock ML model."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0.75]))
    return model


@pytest.fixture
def feature_names() -> list[str]:
    """Sample feature names."""
    return [
        "rsi_14",
        "macd_histogram",
        "sma_50_200_cross",
        "volume_surge",
        "sentiment_score",
    ]


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """Sample feature data."""
    return pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "rsi_14": [75.3],
            "macd_histogram": [0.05],
            "sma_50_200_cross": [1.2],
            "volume_surge": [1.8],
            "sentiment_score": [0.6],
        }
    )


@pytest.fixture
def background_data(feature_names: list[str]) -> pl.DataFrame:
    """Background data for SHAP explainer."""
    np.random.seed(42)
    data = {
        "symbol": ["AAPL"] * 100,
        "rsi_14": np.random.uniform(30, 70, 100).tolist(),
        "macd_histogram": np.random.uniform(-0.1, 0.1, 100).tolist(),
        "sma_50_200_cross": np.random.uniform(-1, 1, 100).tolist(),
        "volume_surge": np.random.uniform(0.5, 2.0, 100).tolist(),
        "sentiment_score": np.random.uniform(-1, 1, 100).tolist(),
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_shap_explainer() -> Mock:
    """Create a mock SHAP explainer."""
    explainer = Mock()
    explainer.expected_value = 0.5
    explainer.shap_values = Mock(
        return_value=np.array([[0.15, 0.08, -0.05, 0.12, -0.03]])
    )
    return explainer


@pytest.fixture
def sample_explanation() -> PredictionExplanation:
    """Sample prediction explanation."""
    return PredictionExplanation(
        symbol="AAPL",
        prediction=0.75,
        base_value=0.5,
        top_features=[
            FeatureContribution(
                feature_name="rsi_14",
                feature_value=75.3,
                contribution=0.15,
                direction="positive",
                importance_rank=1,
                human_readable="rsi_14 = 75.3 (overbought) -> up by 30.0%",
            ),
            FeatureContribution(
                feature_name="volume_surge",
                feature_value=1.8,
                contribution=0.12,
                direction="positive",
                importance_rank=2,
                human_readable="volume_surge = 1.80 (high) -> up by 24.0%",
            ),
            FeatureContribution(
                feature_name="macd_histogram",
                feature_value=0.05,
                contribution=0.08,
                direction="positive",
                importance_rank=3,
                human_readable="macd_histogram = 0.050 (bullish) -> up by 16.0%",
            ),
        ],
        total_positive_contribution=0.35,
        total_negative_contribution=-0.08,
        narrative="The model is strongly bullish on AAPL primarily due to overbought conditions.",
        confidence_factors=["High agreement among top features"],
        generated_at=datetime.now(),
    )


# Schema Tests
class TestFeatureContribution:
    """Tests for FeatureContribution schema."""

    def test_create_feature_contribution(self) -> None:
        """Test creating a feature contribution."""
        contrib = FeatureContribution(
            feature_name="rsi_14",
            feature_value=75.0,
            contribution=0.15,
            direction="positive",
            importance_rank=1,
            human_readable="RSI at 75 pushes prediction up by 15%",
        )

        assert contrib.feature_name == "rsi_14"
        assert contrib.feature_value == 75.0
        assert contrib.contribution == 0.15
        assert contrib.direction == "positive"
        assert contrib.importance_rank == 1

    def test_feature_contribution_validation(self) -> None:
        """Test validation of feature contribution."""
        # Note: Pydantic doesn't enforce string literal validation without custom validator
        # This test demonstrates expected behavior if validation were stricter
        contrib = FeatureContribution(
            feature_name="rsi_14",
            feature_value=75.0,
            contribution=0.15,
            direction="invalid",  # Would be invalid with stricter validation
            importance_rank=1,
            human_readable="test",
        )
        assert contrib.direction == "invalid"  # Currently passes, could be validated


class TestPredictionExplanation:
    """Tests for PredictionExplanation schema."""

    def test_create_prediction_explanation(self, sample_explanation: PredictionExplanation) -> None:
        """Test creating a prediction explanation."""
        assert sample_explanation.symbol == "AAPL"
        assert sample_explanation.prediction == 0.75
        assert sample_explanation.base_value == 0.5
        assert len(sample_explanation.top_features) == 3
        assert sample_explanation.total_positive_contribution == 0.35

    def test_prediction_explanation_serialization(
        self, sample_explanation: PredictionExplanation
    ) -> None:
        """Test serialization of prediction explanation."""
        json_data = sample_explanation.model_dump()

        assert json_data["symbol"] == "AAPL"
        assert json_data["prediction"] == 0.75
        assert len(json_data["top_features"]) == 3


class TestExplanationConfig:
    """Tests for ExplanationConfig schema."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ExplanationConfig()

        assert config.top_k_features == 5
        assert config.include_negative is True
        assert config.generate_narrative is True
        assert config.include_visualization is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ExplanationConfig(
            top_k_features=10,
            include_negative=False,
            generate_narrative=False,
            include_visualization=True,
        )

        assert config.top_k_features == 10
        assert config.include_negative is False
        assert config.generate_narrative is False
        assert config.include_visualization is True

    def test_config_validation(self) -> None:
        """Test config validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExplanationConfig(top_k_features=0)  # Must be >= 1


# SHAPExplainer Tests
class TestSHAPExplainer:
    """Tests for SHAPExplainer."""

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_init_tree_explainer(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
        background_data: pl.DataFrame,
    ) -> None:
        """Test initialization with tree explainer."""
        explainer = SHAPExplainer(mock_model, feature_names, background_data)

        assert explainer.model == mock_model
        assert explainer.feature_names == feature_names
        assert explainer.explainer_type == "tree"
        mock_tree_explainer.assert_called_once()

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_init_without_background_data(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
    ) -> None:
        """Test initialization without background data."""
        explainer = SHAPExplainer(mock_model, feature_names, None)

        assert explainer.background_data is None
        mock_tree_explainer.assert_called_once()

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_calculate_shap_values(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
        sample_features: pl.DataFrame,
    ) -> None:
        """Test SHAP value calculation."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.shap_values = Mock(
            return_value=np.array([[0.1, 0.2, -0.1, 0.3, -0.05]])
        )
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names)
        shap_values = explainer._calculate_shap_values(sample_features)

        assert shap_values.shape == (1, 5)
        assert isinstance(shap_values, np.ndarray)

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_get_top_features(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
    ) -> None:
        """Test getting top features."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names)

        shap_values = [0.15, 0.08, -0.05, 0.12, -0.03]
        feature_values = [75.3, 0.05, 1.2, 1.8, 0.6]

        top_features = explainer._get_top_features(shap_values, feature_values, k=3)

        assert len(top_features) == 3
        assert top_features[0].importance_rank == 1
        assert top_features[0].feature_name == "rsi_14"
        assert top_features[0].contribution == 0.15
        assert top_features[1].feature_name == "volume_surge"
        assert top_features[2].feature_name == "macd_histogram"

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_get_top_features_exclude_negative(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
    ) -> None:
        """Test getting top features excluding negative contributions."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names)

        shap_values = [0.15, 0.08, -0.05, 0.12, -0.03]
        feature_values = [75.3, 0.05, 1.2, 1.8, 0.6]

        top_features = explainer._get_top_features(
            shap_values, feature_values, k=5, include_negative=False
        )

        assert all(f.direction == "positive" for f in top_features)
        assert len(top_features) == 3  # Only 3 positive features

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_format_feature_description_rsi(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
    ) -> None:
        """Test formatting RSI feature description."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names)

        # Overbought
        desc = explainer._format_feature_description("rsi_14", 75.3, 0.15)
        assert "overbought" in desc
        assert "up" in desc

        # Oversold
        desc = explainer._format_feature_description("rsi_14", 25.0, 0.10)
        assert "oversold" in desc

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_format_feature_description_macd(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
    ) -> None:
        """Test formatting MACD feature description."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names)

        # Bullish
        desc = explainer._format_feature_description("macd_histogram", 0.05, 0.08)
        assert "bullish" in desc

        # Bearish
        desc = explainer._format_feature_description("macd_histogram", -0.03, -0.05)
        assert "bearish" in desc

    @patch("signalforge.explanations.narrative.NarrativeGenerator")
    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_explain_prediction(
        self,
        mock_tree_explainer: Mock,
        mock_narrative_gen: Mock,
        mock_model: Mock,
        feature_names: list[str],
        sample_features: pl.DataFrame,
    ) -> None:
        """Test explaining a single prediction."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_explainer_instance.shap_values = Mock(
            return_value=np.array([[0.15, 0.08, -0.05, 0.12, -0.03]])
        )
        mock_tree_explainer.return_value = mock_explainer_instance

        mock_narrative_instance = Mock()
        mock_narrative_instance.generate_narrative = Mock(
            return_value="Bullish prediction for AAPL"
        )
        mock_narrative_gen.return_value = mock_narrative_instance

        explainer = SHAPExplainer(mock_model, feature_names)
        explanation = explainer.explain_prediction(sample_features)

        assert explanation.symbol == "AAPL"
        assert explanation.prediction == 0.75
        assert explanation.base_value == 0.5
        assert len(explanation.top_features) == 5
        assert explanation.narrative == "Bullish prediction for AAPL"

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_explain_prediction_multiple_rows_fails(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
        background_data: pl.DataFrame,
    ) -> None:
        """Test that explaining multiple rows at once fails."""
        explainer = SHAPExplainer(mock_model, feature_names)

        with pytest.raises(ValueError, match="Expected single row"):
            explainer.explain_prediction(background_data[:2])

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_explain_batch(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
        background_data: pl.DataFrame,
    ) -> None:
        """Test explaining batch predictions."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_explainer_instance.shap_values = Mock(
            return_value=np.array([[0.15, 0.08, -0.05, 0.12, -0.03]])
        )
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names)
        explanations = explainer.explain_batch(background_data[:3])

        assert len(explanations) == 3
        assert all(isinstance(exp, PredictionExplanation) for exp in explanations)

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_get_feature_importance(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
        background_data: pl.DataFrame,
    ) -> None:
        """Test getting global feature importance."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_explainer_instance.shap_values = Mock(
            return_value=np.random.rand(100, 5)
        )
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names, background_data)
        importance = explainer.get_feature_importance()

        assert len(importance) == 5
        assert all(name in importance for name in feature_names)
        assert all(isinstance(val, float) for val in importance.values())

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_get_feature_importance_without_background_fails(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
    ) -> None:
        """Test that feature importance requires background data."""
        explainer = SHAPExplainer(mock_model, feature_names, None)

        with pytest.raises(ValueError, match="Background data required"):
            explainer.get_feature_importance()

    @patch("signalforge.explanations.shap_explainer.shap.TreeExplainer")
    def test_compare_predictions(
        self,
        mock_tree_explainer: Mock,
        mock_model: Mock,
        feature_names: list[str],
        sample_features: pl.DataFrame,
    ) -> None:
        """Test comparing two predictions."""
        mock_explainer_instance = Mock()
        mock_explainer_instance.expected_value = 0.5
        mock_explainer_instance.shap_values = Mock(
            return_value=np.array([[0.15, 0.08, -0.05, 0.12, -0.03]])
        )
        mock_tree_explainer.return_value = mock_explainer_instance

        explainer = SHAPExplainer(mock_model, feature_names)

        # Create second sample with different values
        sample_2 = sample_features.clone()
        sample_2 = sample_2.with_columns(pl.lit(80.0).alias("rsi_14"))

        comparison = explainer.compare_predictions(sample_features, sample_2)

        assert "prediction_1" in comparison
        assert "prediction_2" in comparison
        assert "prediction_difference" in comparison
        assert "top_feature_changes" in comparison
        assert len(comparison["top_feature_changes"]) <= 5


# NarrativeGenerator Tests
class TestNarrativeGenerator:
    """Tests for NarrativeGenerator."""

    def test_init(self) -> None:
        """Test narrative generator initialization."""
        generator = NarrativeGenerator()
        assert generator is not None

    def test_determine_sentiment_bullish_strong(self, sample_explanation: PredictionExplanation) -> None:
        """Test determining strong bullish sentiment."""
        generator = NarrativeGenerator()
        sentiment = generator._determine_sentiment(sample_explanation)
        assert "bullish" in sentiment

    def test_determine_sentiment_bearish(self) -> None:
        """Test determining bearish sentiment."""
        generator = NarrativeGenerator()

        bearish_explanation = PredictionExplanation(
            symbol="AAPL",
            prediction=0.3,
            base_value=0.5,
            top_features=[
                FeatureContribution(
                    feature_name="rsi_14",
                    feature_value=25.0,
                    contribution=-0.15,
                    direction="negative",
                    importance_rank=1,
                    human_readable="rsi_14 = 25.0 (oversold) -> down by 30.0%",
                ),
            ],
            total_positive_contribution=0.0,
            total_negative_contribution=-0.2,
            narrative="",
            confidence_factors=[],
            generated_at=datetime.now(),
        )

        sentiment = generator._determine_sentiment(bearish_explanation)
        assert "bearish" in sentiment

    def test_determine_sentiment_neutral(self) -> None:
        """Test determining neutral sentiment."""
        generator = NarrativeGenerator()

        neutral_explanation = PredictionExplanation(
            symbol="AAPL",
            prediction=0.51,
            base_value=0.5,
            top_features=[
                FeatureContribution(
                    feature_name="rsi_14",
                    feature_value=50.0,
                    contribution=0.01,
                    direction="positive",
                    importance_rank=1,
                    human_readable="rsi_14 = 50.0 (neutral) -> up by 2.0%",
                ),
            ],
            total_positive_contribution=0.01,
            total_negative_contribution=0.0,
            narrative="",
            confidence_factors=[],
            generated_at=datetime.now(),
        )

        sentiment = generator._determine_sentiment(neutral_explanation)
        assert sentiment == "neutral"

    def test_select_template(self) -> None:
        """Test template selection."""
        generator = NarrativeGenerator()

        template = generator._select_template(0.75, "bullish_strong")
        assert "strongly bullish" in template

        template = generator._select_template(0.3, "bearish_strong")
        assert "bearish" in template.lower()

    def test_describe_feature_rsi(self) -> None:
        """Test describing RSI feature."""
        generator = NarrativeGenerator()

        contrib = FeatureContribution(
            feature_name="rsi_14",
            feature_value=75.0,
            contribution=0.15,
            direction="positive",
            importance_rank=1,
            human_readable="rsi_14 = 75.0 (overbought) -> up by 30.0%",
        )

        description = generator._describe_feature(contrib)
        assert "overbought" in description or "rsi" in description.lower()

    def test_describe_feature_macd(self) -> None:
        """Test describing MACD feature."""
        generator = NarrativeGenerator()

        contrib = FeatureContribution(
            feature_name="macd_histogram",
            feature_value=0.05,
            contribution=0.08,
            direction="positive",
            importance_rank=2,
            human_readable="macd_histogram = 0.050 (bullish) -> up by 16.0%",
        )

        description = generator._describe_feature(contrib)
        assert "momentum" in description or "macd" in description.lower()

    def test_combine_factors_single(self) -> None:
        """Test combining single factor."""
        generator = NarrativeGenerator()
        result = generator._combine_factors(["factor1"])
        assert result == "factor1"

    def test_combine_factors_two(self) -> None:
        """Test combining two factors."""
        generator = NarrativeGenerator()
        result = generator._combine_factors(["factor1", "factor2"])
        assert result == "factor1 and factor2"

    def test_combine_factors_three(self) -> None:
        """Test combining three factors with oxford comma."""
        generator = NarrativeGenerator()
        result = generator._combine_factors(["factor1", "factor2", "factor3"])
        assert result == "factor1, factor2, and factor3"

    def test_generate_narrative(self, sample_explanation: PredictionExplanation) -> None:
        """Test narrative generation."""
        generator = NarrativeGenerator()
        narrative = generator.generate_narrative(sample_explanation)

        assert len(narrative) > 0
        assert sample_explanation.symbol in narrative
        assert isinstance(narrative, str)

    def test_generate_narrative_with_technical_details(
        self, sample_explanation: PredictionExplanation
    ) -> None:
        """Test narrative generation with technical details."""
        generator = NarrativeGenerator()
        narrative = generator.generate_narrative(sample_explanation, include_technical_details=True)

        assert "Base:" in narrative
        assert "Prediction:" in narrative

    def test_generate_summary(self) -> None:
        """Test summary generation for multiple predictions."""
        generator = NarrativeGenerator()

        explanations = [
            PredictionExplanation(
                symbol=f"SYM{i}",
                prediction=0.6 + i * 0.1,
                base_value=0.5,
                top_features=[
                    FeatureContribution(
                        feature_name="rsi_14",
                        feature_value=60.0 + i * 5,
                        contribution=0.1,
                        direction="positive",
                        importance_rank=1,
                        human_readable="test",
                    ),
                ],
                total_positive_contribution=0.1,
                total_negative_contribution=0.0,
                narrative="",
                confidence_factors=[],
                generated_at=datetime.now(),
            )
            for i in range(3)
        ]

        summary = generator.generate_summary(explanations)

        assert "3 predictions" in summary
        assert "Average prediction:" in summary

    def test_generate_summary_empty(self) -> None:
        """Test summary generation with no predictions."""
        generator = NarrativeGenerator()
        summary = generator.generate_summary([])
        assert "No predictions" in summary


# ExplanationVisualizer Tests
class TestExplanationVisualizer:
    """Tests for ExplanationVisualizer."""

    def test_init(self) -> None:
        """Test visualizer initialization."""
        visualizer = ExplanationVisualizer()
        assert visualizer is not None

    def test_generate_waterfall_data(self, sample_explanation: PredictionExplanation) -> None:
        """Test waterfall chart data generation."""
        visualizer = ExplanationVisualizer()
        waterfall = visualizer.generate_waterfall_data(sample_explanation)

        assert "base_value" in waterfall
        assert "final_prediction" in waterfall
        assert "features" in waterfall
        assert waterfall["base_value"] == 0.5
        assert waterfall["final_prediction"] == 0.75

    def test_waterfall_data_cumulative_values(self, sample_explanation: PredictionExplanation) -> None:
        """Test that waterfall data has correct cumulative values."""
        visualizer = ExplanationVisualizer()
        waterfall = visualizer.generate_waterfall_data(sample_explanation)

        features = waterfall["features"]
        assert len(features) > 0

        # Check first feature starts at base value
        assert features[0]["start"] == waterfall["base_value"]

        # Check cumulative progression
        for i in range(len(features) - 1):
            assert features[i]["end"] == features[i + 1]["start"]

    def test_generate_force_plot_data(self, sample_explanation: PredictionExplanation) -> None:
        """Test force plot data generation."""
        visualizer = ExplanationVisualizer()
        force_plot = visualizer.generate_force_plot_data(sample_explanation)

        assert "base_value" in force_plot
        assert "prediction" in force_plot
        assert "positive_forces" in force_plot
        assert "negative_forces" in force_plot
        assert "total_positive" in force_plot
        assert "total_negative" in force_plot

    def test_force_plot_separates_positive_negative(
        self, sample_explanation: PredictionExplanation
    ) -> None:
        """Test that force plot separates positive and negative contributions."""
        visualizer = ExplanationVisualizer()
        force_plot = visualizer.generate_force_plot_data(sample_explanation)

        # All positive forces should have positive contribution
        for force in force_plot["positive_forces"]:
            assert force["contribution"] > 0

        # All negative forces should have positive magnitude (abs value)
        for force in force_plot["negative_forces"]:
            assert force["contribution"] > 0  # Magnitude, not signed value

    def test_generate_summary_plot_data(self) -> None:
        """Test summary plot data generation."""
        visualizer = ExplanationVisualizer()

        explanations = [
            PredictionExplanation(
                symbol=f"SYM{i}",
                prediction=0.6 + i * 0.1,
                base_value=0.5,
                top_features=[
                    FeatureContribution(
                        feature_name="rsi_14",
                        feature_value=60.0,
                        contribution=0.1,
                        direction="positive",
                        importance_rank=1,
                        human_readable="test",
                    ),
                    FeatureContribution(
                        feature_name="macd",
                        feature_value=0.05,
                        contribution=0.05,
                        direction="positive",
                        importance_rank=2,
                        human_readable="test",
                    ),
                ],
                total_positive_contribution=0.15,
                total_negative_contribution=0.0,
                narrative="",
                confidence_factors=[],
                generated_at=datetime.now(),
            )
            for i in range(3)
        ]

        summary = visualizer.generate_summary_plot_data(explanations)

        assert summary["num_predictions"] == 3
        assert "features" in summary
        assert "symbols" in summary
        assert len(summary["features"]) > 0

    def test_summary_plot_empty_list(self) -> None:
        """Test summary plot with empty list."""
        visualizer = ExplanationVisualizer()
        summary = visualizer.generate_summary_plot_data([])

        assert summary["num_predictions"] == 0
        assert summary["features"] == []

    def test_generate_feature_importance_chart(self) -> None:
        """Test feature importance chart generation."""
        visualizer = ExplanationVisualizer()

        importance = {
            "rsi_14": 0.25,
            "macd": 0.18,
            "volume": 0.15,
            "sma_cross": 0.12,
            "sentiment": 0.10,
        }

        chart = visualizer.generate_feature_importance_chart(importance, top_k=3)

        assert chart["top_k"] == 3
        assert chart["total_features"] == 5
        assert len(chart["features"]) == 3
        assert chart["features"][0]["name"] == "rsi_14"  # Highest importance

    def test_feature_importance_chart_ranking(self) -> None:
        """Test that feature importance chart is properly ranked."""
        visualizer = ExplanationVisualizer()

        importance = {
            "feature_a": 0.10,
            "feature_b": 0.25,
            "feature_c": 0.15,
        }

        chart = visualizer.generate_feature_importance_chart(importance)

        # Should be sorted by importance
        assert chart["features"][0]["importance"] >= chart["features"][1]["importance"]
        assert chart["features"][1]["importance"] >= chart["features"][2]["importance"]

        # Check ranks
        assert chart["features"][0]["rank"] == 1
        assert chart["features"][1]["rank"] == 2

    def test_export_to_html(self, sample_explanation: PredictionExplanation, tmp_path: Path) -> None:
        """Test HTML export."""
        visualizer = ExplanationVisualizer()

        output_path = tmp_path / "explanation.html"
        result_path = visualizer.export_to_html(sample_explanation, str(output_path))

        assert Path(result_path).exists()
        assert Path(result_path).suffix == ".html"

        # Check content
        content = Path(result_path).read_text()
        assert sample_explanation.symbol in content
        assert str(sample_explanation.prediction) in content

    def test_export_to_html_creates_directory(
        self, sample_explanation: PredictionExplanation, tmp_path: Path
    ) -> None:
        """Test that HTML export creates directory if needed."""
        visualizer = ExplanationVisualizer()

        output_path = tmp_path / "subdir" / "explanation.html"
        result_path = visualizer.export_to_html(sample_explanation, str(output_path))

        assert Path(result_path).exists()
        assert Path(result_path).parent.exists()
