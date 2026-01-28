"""Tests for SHAP explainability module.

This module tests the explainer functionality including:
- Feature importance calculation
- Explanation generation for single and batch predictions
- Text summary generation
- Visualization data preparation
- Fallback to permutation importance
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest

from signalforge.ml.inference.explainer import (
    BaseExplainer,
    ExplanationResult,
    ExplainerConfig,
    FeatureImportance,
    ModelExplainer,
    generate_explanation_text,
    plot_summary,
    plot_waterfall,
)


class MockModel:
    """Mock model for testing that implements predict method."""

    def __init__(self, return_value: float | list[float] = 0.65) -> None:
        """Initialize mock model.

        Args:
            return_value: Value or list of values to return from predict
        """
        self.return_value = return_value
        self.predict_called = False
        self.predict_call_count = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock predict method.

        Args:
            X: Input features

        Returns:
            Array of predictions
        """
        self.predict_called = True
        self.predict_call_count += 1

        if isinstance(self.return_value, list):
            return np.array(self.return_value)
        else:
            return np.full(X.shape[0], self.return_value)


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "rsi_14": [65.0, 45.0, 55.0],
            "macd": [0.5, -0.3, 0.1],
            "sma_20": [100.5, 99.2, 100.0],
            "volume": [1000000.0, 800000.0, 900000.0],
        }
    )


@pytest.fixture
def single_row_dataframe() -> pl.DataFrame:
    """Create single row DataFrame for testing."""
    return pl.DataFrame(
        {
            "rsi_14": [65.0],
            "macd": [0.5],
            "sma_20": [100.5],
            "volume": [1000000.0],
        }
    )


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock model for testing."""
    return MockModel(return_value=0.65)


@pytest.fixture
def explainer_config() -> ExplainerConfig:
    """Create explainer configuration for testing."""
    return ExplainerConfig(method="kernel", n_samples=10, max_features=3)


class TestFeatureImportance:
    """Tests for FeatureImportance dataclass."""

    def test_feature_importance_creation(self) -> None:
        """Test creating FeatureImportance instance."""
        fi = FeatureImportance(feature="rsi_14", importance=0.08, direction="positive")

        assert fi.feature == "rsi_14"
        assert fi.importance == 0.08
        assert fi.direction == "positive"

    def test_feature_importance_negative_direction(self) -> None:
        """Test FeatureImportance with negative direction."""
        fi = FeatureImportance(feature="macd", importance=0.05, direction="negative")

        assert fi.feature == "macd"
        assert fi.importance == 0.05
        assert fi.direction == "negative"


class TestExplanationResult:
    """Tests for ExplanationResult dataclass."""

    def test_explanation_result_creation(self) -> None:
        """Test creating ExplanationResult instance."""
        contributions = [
            FeatureImportance("rsi_14", 0.08, "positive"),
            FeatureImportance("macd", 0.05, "positive"),
        ]

        result = ExplanationResult(
            prediction=0.65,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=["rsi_14", "macd"],
            summary_text="Test explanation",
        )

        assert result.prediction == 0.65
        assert result.base_value == 0.50
        assert len(result.feature_contributions) == 2
        assert result.top_features == ["rsi_14", "macd"]
        assert result.summary_text == "Test explanation"


class TestExplainerConfig:
    """Tests for ExplainerConfig dataclass."""

    def test_explainer_config_defaults(self) -> None:
        """Test ExplainerConfig default values."""
        config = ExplainerConfig()

        assert config.method == "kernel"
        assert config.n_samples == 100
        assert config.max_features == 10

    def test_explainer_config_custom(self) -> None:
        """Test ExplainerConfig with custom values."""
        config = ExplainerConfig(method="tree", n_samples=50, max_features=5)

        assert config.method == "tree"
        assert config.n_samples == 50
        assert config.max_features == 5


class TestBaseExplainer:
    """Tests for BaseExplainer abstract class."""

    def test_base_explainer_is_abstract(self) -> None:
        """Test that BaseExplainer cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseExplainer()  # type: ignore[abstract]

    def test_base_explainer_requires_implementation(self) -> None:
        """Test that subclasses must implement abstract methods."""

        class IncompleteExplainer(BaseExplainer):
            pass

        with pytest.raises(TypeError):
            IncompleteExplainer()  # type: ignore[abstract]


class TestModelExplainer:
    """Tests for ModelExplainer class."""

    def test_model_explainer_initialization(self, explainer_config: ExplainerConfig) -> None:
        """Test ModelExplainer initialization."""
        explainer = ModelExplainer(explainer_config)

        assert explainer.config.method == "kernel"
        assert explainer.config.n_samples == 10
        assert explainer.config.max_features == 3
        assert explainer._explainer is None
        assert explainer._background_data is None

    def test_model_explainer_default_config(self) -> None:
        """Test ModelExplainer with default configuration."""
        explainer = ModelExplainer()

        assert explainer.config.method == "kernel"
        assert explainer.config.n_samples == 100
        assert explainer.config.max_features == 10

    def test_initialize_explainer_kernel(
        self,
        mock_model: MockModel,
        sample_dataframe: pl.DataFrame,
        explainer_config: ExplainerConfig,
    ) -> None:
        """Test SHAP explainer initialization with KernelExplainer."""
        # Create mock shap module
        mock_shap = MagicMock()
        mock_kernel_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_kernel_explainer

        with patch.dict(sys.modules, {"shap": mock_shap}):
            explainer = ModelExplainer(explainer_config)
            explainer._initialize_explainer(mock_model, sample_dataframe)

            # Check that background data was set
            assert explainer._background_data is not None
            assert explainer._explainer is not None

            # Check that KernelExplainer was called
            mock_shap.KernelExplainer.assert_called_once()

    @patch("signalforge.ml.inference.explainer.shap")
    def test_explain_single_prediction(
        self,
        mock_shap: MagicMock,
        mock_model: MockModel,
        single_row_dataframe: pl.DataFrame,
        explainer_config: ExplainerConfig,
    ) -> None:
        """Test explaining a single prediction."""
        # Setup mock SHAP explainer
        mock_explainer = MagicMock()
        mock_explainer.expected_value = 0.50
        mock_shap.KernelExplainer.return_value = mock_explainer

        # Mock SHAP values
        mock_shap_values = np.array([0.08, 0.05, 0.02, 0.01])

        def mock_shap_values_func(X: np.ndarray) -> np.ndarray:
            return mock_shap_values

        mock_explainer.shap_values = mock_shap_values_func

        explainer = ModelExplainer(explainer_config)
        result = explainer.explain(mock_model, single_row_dataframe)

        # Verify result structure
        assert isinstance(result, ExplanationResult)
        assert result.prediction == 0.65
        assert result.base_value == 0.50
        assert len(result.feature_contributions) == 4
        assert len(result.top_features) == 3  # max_features is 3
        assert result.summary_text != ""

        # Verify feature contributions are sorted
        importances = [fc.importance for fc in result.feature_contributions]
        assert importances == sorted(importances, reverse=True)

    @patch("signalforge.ml.inference.explainer.shap")
    def test_explain_batch_predictions(
        self,
        mock_shap: MagicMock,
        mock_model: MockModel,
        sample_dataframe: pl.DataFrame,
        explainer_config: ExplainerConfig,
    ) -> None:
        """Test explaining batch predictions."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_explainer.expected_value = 0.50
        mock_shap.KernelExplainer.return_value = mock_explainer

        # Mock SHAP values for 3 samples, 4 features
        mock_shap_values = np.array(
            [[0.08, 0.05, 0.02, 0.01], [0.03, -0.04, 0.01, 0.02], [0.06, 0.03, -0.02, 0.01]]
        )

        def mock_shap_values_func(X: np.ndarray) -> np.ndarray:
            return mock_shap_values

        mock_explainer.shap_values = mock_shap_values_func

        # Update mock model to return multiple predictions
        mock_model.return_value = [0.65, 0.55, 0.60]

        explainer = ModelExplainer(explainer_config)
        results = explainer.explain_batch(mock_model, sample_dataframe)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(r, ExplanationResult) for r in results)

        # Verify each result
        for result in results:
            assert result.prediction >= 0
            assert result.base_value == 0.50
            assert len(result.feature_contributions) == 4
            assert len(result.top_features) <= 3
            assert result.summary_text != ""

    @patch("signalforge.ml.inference.explainer.shap")
    def test_get_feature_importance(
        self,
        mock_shap: MagicMock,
        mock_model: MockModel,
        sample_dataframe: pl.DataFrame,
        explainer_config: ExplainerConfig,
    ) -> None:
        """Test calculating global feature importance."""
        # Setup mock
        mock_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_explainer

        # Mock SHAP values
        mock_shap_values = np.array(
            [[0.08, 0.05, 0.02, 0.01], [0.03, -0.04, 0.01, 0.02], [0.06, 0.03, -0.02, 0.01]]
        )

        def mock_shap_values_func(X: np.ndarray) -> np.ndarray:
            return mock_shap_values

        mock_explainer.shap_values = mock_shap_values_func

        explainer = ModelExplainer(explainer_config)
        importances = explainer.get_feature_importance(mock_model, sample_dataframe)

        # Verify results
        assert len(importances) == 4
        assert all(isinstance(imp, FeatureImportance) for imp in importances)

        # Verify sorted by importance
        importance_values = [imp.importance for imp in importances]
        assert importance_values == sorted(importance_values, reverse=True)

    def test_explain_empty_dataframe(
        self, mock_model: MockModel, explainer_config: ExplainerConfig
    ) -> None:
        """Test explaining with empty DataFrame raises error."""
        empty_df = pl.DataFrame()
        explainer = ModelExplainer(explainer_config)

        with pytest.raises(ValueError, match="X must contain exactly one row"):
            explainer.explain(mock_model, empty_df)

    def test_explain_multiple_rows_raises_error(
        self, mock_model: MockModel, sample_dataframe: pl.DataFrame, explainer_config: ExplainerConfig
    ) -> None:
        """Test that explain with multiple rows raises error."""
        explainer = ModelExplainer(explainer_config)

        with pytest.raises(ValueError, match="exactly one row"):
            explainer.explain(mock_model, sample_dataframe)

    def test_explain_batch_empty_dataframe(
        self, mock_model: MockModel, explainer_config: ExplainerConfig
    ) -> None:
        """Test batch explaining with empty DataFrame raises error."""
        empty_df = pl.DataFrame()
        explainer = ModelExplainer(explainer_config)

        with pytest.raises(ValueError, match="cannot be empty"):
            explainer.explain_batch(mock_model, empty_df)

    def test_get_feature_importance_empty_dataframe(
        self, mock_model: MockModel, explainer_config: ExplainerConfig
    ) -> None:
        """Test getting feature importance with empty DataFrame raises error."""
        empty_df = pl.DataFrame()
        explainer = ModelExplainer(explainer_config)

        with pytest.raises(ValueError, match="cannot be empty"):
            explainer.get_feature_importance(mock_model, empty_df)

    def test_fallback_permutation_importance(
        self, mock_model: MockModel, sample_dataframe: pl.DataFrame, explainer_config: ExplainerConfig
    ) -> None:
        """Test fallback to permutation importance when SHAP fails."""
        explainer = ModelExplainer(explainer_config)

        # Call fallback method directly
        importances = explainer._fallback_permutation_importance(mock_model, sample_dataframe)

        # Verify results
        assert len(importances) == 4
        assert all(isinstance(imp, FeatureImportance) for imp in importances)

        # Verify sorted
        importance_values = [imp.importance for imp in importances]
        assert importance_values == sorted(importance_values, reverse=True)

        # Verify model was called
        assert mock_model.predict_called


class TestGenerateExplanationText:
    """Tests for text generation functionality."""

    def test_generate_explanation_text_positive(self) -> None:
        """Test generating explanation text for positive prediction."""
        contributions = [
            FeatureImportance("rsi_14", 0.08, "positive"),
            FeatureImportance("macd", 0.05, "positive"),
        ]

        result = ExplanationResult(
            prediction=0.65,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=["rsi_14", "macd"],
            summary_text="",
        )

        text = generate_explanation_text(result)

        # Verify text contains key information
        assert "0.65" in text
        assert "0.50" in text
        assert "rsi_14" in text
        assert "macd" in text
        assert "POSITIVE" in text

    def test_generate_explanation_text_negative(self) -> None:
        """Test generating explanation text for negative prediction."""
        contributions = [
            FeatureImportance("rsi_14", 0.08, "negative"),
            FeatureImportance("macd", 0.05, "negative"),
        ]

        result = ExplanationResult(
            prediction=0.35,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=["rsi_14", "macd"],
            summary_text="",
        )

        text = generate_explanation_text(result)

        # Verify text contains key information
        assert "0.35" in text
        assert "0.50" in text
        assert "rsi_14" in text
        assert "NEGATIVE" in text

    def test_generate_explanation_text_neutral(self) -> None:
        """Test generating explanation text for neutral prediction."""
        contributions = [
            FeatureImportance("rsi_14", 0.001, "positive"),
        ]

        result = ExplanationResult(
            prediction=0.50,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=["rsi_14"],
            summary_text="",
        )

        text = generate_explanation_text(result)

        # Verify neutral sentiment
        assert "neutral" in text.lower()

    def test_generate_explanation_text_with_financial_indicators(self) -> None:
        """Test that financial indicators get proper context."""
        contributions = [
            FeatureImportance("rsi_14", 0.08, "positive"),
            FeatureImportance("macd_signal", 0.05, "positive"),
            FeatureImportance("sma_20", 0.03, "negative"),
            FeatureImportance("volume_avg", 0.02, "positive"),
            FeatureImportance("bb_upper", 0.01, "positive"),
        ]

        result = ExplanationResult(
            prediction=0.65,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=["rsi_14", "macd_signal", "sma_20", "volume_avg", "bb_upper"],
            summary_text="",
        )

        text = generate_explanation_text(result)

        # Verify financial context is included
        assert "RSI" in text
        assert "MACD" in text
        assert "Moving average" in text or "trend" in text.lower()


class TestVisualizationHelpers:
    """Tests for visualization helper functions."""

    def test_plot_waterfall(self) -> None:
        """Test waterfall plot data generation."""
        contributions = [
            FeatureImportance("rsi_14", 0.08, "positive"),
            FeatureImportance("macd", 0.05, "positive"),
            FeatureImportance("sma_20", 0.03, "negative"),
        ]

        result = ExplanationResult(
            prediction=0.65,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=["rsi_14", "macd", "sma_20"],
            summary_text="Test",
        )

        data = plot_waterfall(result)

        # Verify structure
        assert "features" in data
        assert "values" in data
        assert "base_value" in data
        assert "prediction" in data

        # Verify content
        assert len(data["features"]) == 3
        assert len(data["values"]) == 3
        assert data["base_value"] == 0.50
        assert data["prediction"] == 0.65

        # Verify signed values
        assert data["values"][0] > 0  # positive contribution
        assert data["values"][1] > 0  # positive contribution
        assert data["values"][2] < 0  # negative contribution

    def test_plot_waterfall_limits_features(self) -> None:
        """Test that waterfall plot limits to top 10 features."""
        contributions = [
            FeatureImportance(f"feature_{i}", 1.0 / (i + 1), "positive") for i in range(15)
        ]

        result = ExplanationResult(
            prediction=0.65,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=[f"feature_{i}" for i in range(15)],
            summary_text="Test",
        )

        data = plot_waterfall(result)

        # Should limit to 10 features
        assert len(data["features"]) == 10
        assert len(data["values"]) == 10

    def test_plot_summary_empty_list(self) -> None:
        """Test summary plot with empty explanations list."""
        data = plot_summary([])

        assert data["features"] == []
        assert data["importances"] == []
        assert data["shap_values"] == []

    def test_plot_summary_single_explanation(self) -> None:
        """Test summary plot with single explanation."""
        contributions = [
            FeatureImportance("rsi_14", 0.08, "positive"),
            FeatureImportance("macd", 0.05, "positive"),
        ]

        result = ExplanationResult(
            prediction=0.65,
            base_value=0.50,
            feature_contributions=contributions,
            top_features=["rsi_14", "macd"],
            summary_text="Test",
        )

        data = plot_summary([result])

        # Verify structure
        assert "features" in data
        assert "importances" in data
        assert "shap_values" in data

        # Verify content
        assert len(data["features"]) == 2
        assert len(data["importances"]) == 2
        assert len(data["shap_values"]) == 1
        assert len(data["shap_values"][0]) == 2

    def test_plot_summary_multiple_explanations(self) -> None:
        """Test summary plot with multiple explanations."""
        results = []
        for i in range(3):
            contributions = [
                FeatureImportance("rsi_14", 0.08 - i * 0.01, "positive"),
                FeatureImportance("macd", 0.05 + i * 0.01, "negative"),
            ]

            result = ExplanationResult(
                prediction=0.65 - i * 0.05,
                base_value=0.50,
                feature_contributions=contributions,
                top_features=["rsi_14", "macd"],
                summary_text="Test",
            )
            results.append(result)

        data = plot_summary(results)

        # Verify structure
        assert len(data["features"]) == 2
        assert len(data["importances"]) == 2
        assert len(data["shap_values"]) == 3
        assert all(len(row) == 2 for row in data["shap_values"])

        # Verify importances are positive
        assert all(imp >= 0 for imp in data["importances"])


class TestIntegration:
    """Integration tests for the explainer module."""

    @patch("signalforge.ml.inference.explainer.shap")
    def test_end_to_end_single_explanation(
        self, mock_shap: MagicMock, single_row_dataframe: pl.DataFrame
    ) -> None:
        """Test complete workflow for single prediction explanation."""
        # Setup
        mock_model = MockModel(return_value=0.65)
        mock_explainer = MagicMock()
        mock_explainer.expected_value = 0.50
        mock_shap.KernelExplainer.return_value = mock_explainer

        mock_shap_values = np.array([0.08, 0.05, 0.02, 0.01])
        mock_explainer.shap_values = Mock(return_value=mock_shap_values)

        # Execute
        config = ExplainerConfig(method="kernel", n_samples=10, max_features=3)
        explainer = ModelExplainer(config)
        result = explainer.explain(mock_model, single_row_dataframe)

        # Verify
        assert result.prediction == 0.65
        assert result.base_value == 0.50
        assert len(result.feature_contributions) == 4
        assert len(result.top_features) == 3

        # Generate text
        text = generate_explanation_text(result)
        assert text != ""
        assert "0.65" in text

        # Generate waterfall data
        waterfall_data = plot_waterfall(result)
        assert len(waterfall_data["features"]) == 4
        assert waterfall_data["prediction"] == 0.65

    @patch("signalforge.ml.inference.explainer.shap")
    def test_end_to_end_batch_explanation(
        self, mock_shap: MagicMock, sample_dataframe: pl.DataFrame
    ) -> None:
        """Test complete workflow for batch prediction explanation."""
        # Setup
        mock_model = MockModel(return_value=[0.65, 0.55, 0.60])
        mock_explainer = MagicMock()
        mock_explainer.expected_value = 0.50
        mock_shap.KernelExplainer.return_value = mock_explainer

        mock_shap_values = np.array(
            [[0.08, 0.05, 0.02, 0.01], [0.03, -0.04, 0.01, 0.02], [0.06, 0.03, -0.02, 0.01]]
        )
        mock_explainer.shap_values = Mock(return_value=mock_shap_values)

        # Execute
        config = ExplainerConfig(method="kernel", n_samples=10, max_features=3)
        explainer = ModelExplainer(config)
        results = explainer.explain_batch(mock_model, sample_dataframe)

        # Verify
        assert len(results) == 3

        # Generate summary data
        summary_data = plot_summary(results)
        assert len(summary_data["features"]) == 4
        assert len(summary_data["shap_values"]) == 3
