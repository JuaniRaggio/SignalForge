"""Tests for sentiment analysis module.

This module tests the FinBERT-based sentiment analyzer with mocked models
to avoid requiring actual model downloads during testing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="module")
def mock_torch_module() -> Iterator[MagicMock]:
    """Create a mock torch module that can be used in tests.

    This fixture properly isolates the mock to prevent interference
    with other test modules that use scipy/torch.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.empty_cache = MagicMock()

    # Store original modules
    original_modules: dict[str, Any] = {}
    modules_to_mock = [
        "torch",
        "torch.cuda",
        "torch.backends",
        "torch.backends.mps",
        "transformers",
    ]

    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]

    # Install mocks
    sys.modules["torch"] = mock_torch
    sys.modules["torch.cuda"] = mock_torch.cuda
    sys.modules["torch.backends"] = mock_torch.backends
    sys.modules["torch.backends.mps"] = mock_torch.backends.mps

    mock_transformers = MagicMock()
    sys.modules["transformers"] = mock_transformers

    yield mock_torch

    # Restore original modules
    for mod_name in modules_to_mock:
        if mod_name in original_modules:
            sys.modules[mod_name] = original_modules[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]


@pytest.fixture
def sentiment_module(mock_torch_module: MagicMock) -> Any:
    """Import the sentiment module with mocked dependencies."""
    # Clear any cached imports of the sentiment module
    modules_to_clear = [k for k in list(sys.modules.keys()) if "signalforge.nlp.sentiment" in k]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    # Also clear the parent nlp module to force reimport
    if "signalforge.nlp" in sys.modules:
        del sys.modules["signalforge.nlp"]

    # Import fresh with mocked dependencies
    from signalforge.nlp import sentiment

    return sentiment


@pytest.fixture
def mock_transformers() -> MagicMock:
    """Get the mock transformers module."""
    return sys.modules["transformers"]


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_valid_sentiment_result(self, sentiment_module: Any) -> None:
        """Test creation of valid sentiment result."""
        SentimentResult = sentiment_module.SentimentResult
        result = SentimentResult(
            text="Test text",
            label="positive",
            confidence=0.95,
            scores={"positive": 0.95, "negative": 0.03, "neutral": 0.02},
        )

        assert result.text == "Test text"
        assert result.label == "positive"
        assert result.confidence == 0.95
        assert result.scores["positive"] == 0.95

    def test_invalid_confidence_range(self, sentiment_module: Any) -> None:
        """Test that invalid confidence values raise ValueError."""
        SentimentResult = sentiment_module.SentimentResult
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SentimentResult(
                text="Test",
                label="positive",
                confidence=1.5,
                scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            )

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SentimentResult(
                text="Test",
                label="positive",
                confidence=-0.1,
                scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            )

    def test_invalid_label(self, sentiment_module: Any) -> None:
        """Test that invalid labels raise ValueError."""
        SentimentResult = sentiment_module.SentimentResult
        with pytest.raises(ValueError, match="Invalid label"):
            SentimentResult(
                text="Test",
                label="invalid",  # type: ignore[arg-type]
                confidence=0.8,
                scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            )

    def test_invalid_scores_keys(self, sentiment_module: Any) -> None:
        """Test that missing or extra score keys raise ValueError."""
        SentimentResult = sentiment_module.SentimentResult
        with pytest.raises(ValueError, match="Scores must contain exactly"):
            SentimentResult(
                text="Test",
                label="positive",
                confidence=0.8,
                scores={"positive": 0.8, "negative": 0.2},  # Missing neutral
            )

    def test_invalid_score_values(self, sentiment_module: Any) -> None:
        """Test that score values outside 0-1 range raise ValueError."""
        SentimentResult = sentiment_module.SentimentResult
        with pytest.raises(ValueError, match="Score for .* must be between 0.0 and 1.0"):
            SentimentResult(
                text="Test",
                label="positive",
                confidence=0.8,
                scores={"positive": 1.5, "negative": 0.1, "neutral": 0.1},
            )


class TestSentimentConfig:
    """Tests for SentimentConfig dataclass."""

    def test_default_config(self, sentiment_module: Any) -> None:
        """Test default configuration values."""
        SentimentConfig = sentiment_module.SentimentConfig
        config = SentimentConfig()

        assert config.model_name == "ProsusAI/finbert"
        assert config.device == "auto"
        assert config.batch_size == 16
        assert config.max_length == 512
        assert config.cache_model is True
        assert config.preprocess_text is False
        assert config.temperature == 1.0

    def test_custom_config(self, sentiment_module: Any) -> None:
        """Test custom configuration values."""
        SentimentConfig = sentiment_module.SentimentConfig
        config = SentimentConfig(
            model_name="custom/model",
            device="cpu",
            batch_size=32,
            max_length=256,
            cache_model=False,
            preprocess_text=True,
            temperature=0.5,
        )

        assert config.model_name == "custom/model"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.max_length == 256
        assert config.cache_model is False
        assert config.preprocess_text is True
        assert config.temperature == 0.5

    def test_invalid_batch_size(self, sentiment_module: Any) -> None:
        """Test that invalid batch size raises ValueError."""
        SentimentConfig = sentiment_module.SentimentConfig
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SentimentConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            SentimentConfig(batch_size=-1)

    def test_invalid_max_length(self, sentiment_module: Any) -> None:
        """Test that invalid max length raises ValueError."""
        SentimentConfig = sentiment_module.SentimentConfig
        with pytest.raises(ValueError, match="max_length must be positive"):
            SentimentConfig(max_length=0)

        with pytest.raises(ValueError, match="max_length must be positive"):
            SentimentConfig(max_length=-1)

    def test_invalid_device(self, sentiment_module: Any) -> None:
        """Test that invalid device raises ValueError."""
        SentimentConfig = sentiment_module.SentimentConfig
        with pytest.raises(ValueError, match="device must be one of"):
            SentimentConfig(device="invalid")

    def test_invalid_temperature(self, sentiment_module: Any) -> None:
        """Test that invalid temperature raises ValueError."""
        SentimentConfig = sentiment_module.SentimentConfig
        with pytest.raises(ValueError, match="temperature must be positive"):
            SentimentConfig(temperature=0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            SentimentConfig(temperature=-1)


class TestBaseSentimentAnalyzer:
    """Tests for BaseSentimentAnalyzer abstract class."""

    def test_cannot_instantiate(self, sentiment_module: Any) -> None:
        """Test that abstract class cannot be instantiated."""
        BaseSentimentAnalyzer = sentiment_module.BaseSentimentAnalyzer
        with pytest.raises(TypeError):
            BaseSentimentAnalyzer()  # type: ignore[abstract]


class TestFinBERTSentimentAnalyzer:
    """Tests for FinBERTSentimentAnalyzer class."""

    def test_initialization(self, sentiment_module: Any) -> None:
        """Test analyzer initialization."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        config = SentimentConfig(device="cpu")
        analyzer = FinBERTSentimentAnalyzer(config)

        assert analyzer.model_name == "ProsusAI/finbert"
        assert analyzer._config.device == "cpu"
        assert analyzer._pipeline is None  # Lazy loading

    def test_model_name_property(self, sentiment_module: Any) -> None:
        """Test model_name property."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        config = SentimentConfig(model_name="test/model")
        analyzer = FinBERTSentimentAnalyzer(config)

        assert analyzer.model_name == "test/model"

    def test_get_device_cpu(self, sentiment_module: Any) -> None:
        """Test device selection for CPU."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        config = SentimentConfig(device="cpu")
        analyzer = FinBERTSentimentAnalyzer(config)

        device = analyzer._get_device()
        assert device == -1

    def test_get_device_cuda_available(
        self, sentiment_module: Any, mock_torch_module: MagicMock
    ) -> None:
        """Test device selection when CUDA is available."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        with patch.object(mock_torch_module.cuda, "is_available", return_value=True):
            config = SentimentConfig(device="auto")
            analyzer = FinBERTSentimentAnalyzer(config)

            device = analyzer._get_device()
            assert device == 0

    def test_get_device_auto_fallback_cpu(
        self, sentiment_module: Any, mock_torch_module: MagicMock
    ) -> None:
        """Test device selection falls back to CPU when GPU unavailable."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        with (
            patch.object(mock_torch_module.cuda, "is_available", return_value=False),
            patch.object(mock_torch_module.backends.mps, "is_available", return_value=False),
        ):
            config = SentimentConfig(device="auto")
            analyzer = FinBERTSentimentAnalyzer(config)

            device = analyzer._get_device()
            assert device == -1

    def test_analyze_empty_text(self, sentiment_module: Any) -> None:
        """Test that analyzing empty text raises ValueError."""
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.analyze("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.analyze("   ")

    def test_analyze_single_text(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test analyzing single text."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.95}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu")
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("Strong quarterly earnings beat expectations.")

            assert result.text == "Strong quarterly earnings beat expectations."
            assert result.label == "positive"
            assert result.confidence == 0.95
            assert "positive" in result.scores
            assert "negative" in result.scores
            assert "neutral" in result.scores

    def test_analyze_negative_sentiment(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test analyzing text with negative sentiment."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "negative", "score": 0.88}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("Company missed earnings expectations.")

            assert result.label == "negative"
            assert result.confidence == 0.88

    def test_analyze_neutral_sentiment(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test analyzing text with neutral sentiment."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.75}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("The company reported quarterly results.")

            assert result.label == "neutral"
            assert result.confidence == 0.75

    def test_analyze_batch_empty_list(self, sentiment_module: Any) -> None:
        """Test that analyzing empty batch raises ValueError."""
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            analyzer.analyze_batch([])

    def test_analyze_batch(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test batch sentiment analysis."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "positive", "score": 0.92},
            {"label": "negative", "score": 0.85},
        ]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", batch_size=2, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            texts = [
                "Revenue exceeded expectations.",
                "Losses widened this quarter.",
            ]
            results = analyzer.analyze_batch(texts)

            assert len(results) == 2
            assert results[0].label == "positive"
            assert results[0].confidence == 0.92
            assert results[1].label == "negative"
            assert results[1].confidence == 0.85

    def test_analyze_batch_with_empty_texts(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test batch analysis with some empty texts."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu")
            analyzer = FinBERTSentimentAnalyzer(config)

            texts = ["Good earnings.", "", "   "]
            results = analyzer.analyze_batch(texts)

            assert len(results) == 3
            assert results[0].label == "positive"
            assert results[1].label == "neutral"
            assert results[2].label == "neutral"

    def test_analyze_batch_multiple_batches(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test batch analysis with multiple batches."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = [
            [{"label": "positive", "score": 0.90}, {"label": "negative", "score": 0.85}],
            [{"label": "neutral", "score": 0.70}],
        ]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", batch_size=2, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            texts = ["Text 1", "Text 2", "Text 3"]
            results = analyzer.analyze_batch(texts)

            assert len(results) == 3
            assert mock_pipeline.call_count == 2

    def test_analyze_with_preprocessing(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test analysis with text preprocessing enabled."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", preprocess_text=True, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            text = "  Strong earnings!  http://example.com  "
            result = analyzer.analyze(text)

            assert result.label == "positive"
            call_args = mock_pipeline.call_args[0][0]
            assert "http://example.com" not in call_args

    def test_model_caching(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test that model is loaded only once when caching is enabled."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(
            mock_transformers, "pipeline", return_value=mock_pipeline
        ) as mock_factory:
            config = SentimentConfig(device="cpu", cache_model=True)
            analyzer = FinBERTSentimentAnalyzer(config)

            analyzer._model_cache.clear()

            analyzer.analyze("Text 1")
            first_call_count = mock_factory.call_count

            analyzer.analyze("Text 2")
            second_call_count = mock_factory.call_count

            assert second_call_count == first_call_count

    def test_model_load_failure_import_error(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test handling of model loading failure due to missing dependencies."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        with patch.object(
            mock_transformers, "pipeline", side_effect=ImportError("transformers not installed")
        ):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            with pytest.raises(RuntimeError, match="transformers or torch not installed"):
                analyzer.analyze("Test text")

    def test_model_load_failure_general_error(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test handling of general model loading failure."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        with patch.object(
            mock_transformers, "pipeline", side_effect=Exception("Model download failed")
        ):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            with pytest.raises(RuntimeError, match="Failed to load model"):
                analyzer.analyze("Test text")

    def test_cleanup_gpu_memory(self, sentiment_module: Any, mock_torch_module: MagicMock) -> None:
        """Test GPU memory cleanup."""
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        with patch.object(mock_torch_module.cuda, "is_available", return_value=True):
            analyzer = FinBERTSentimentAnalyzer()
            analyzer._pipeline = MagicMock()

            analyzer.cleanup()

            mock_torch_module.cuda.empty_cache.assert_called()
            assert analyzer._pipeline is None

    def test_cleanup_no_gpu(self, sentiment_module: Any) -> None:
        """Test cleanup when no GPU is used."""
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock()

        analyzer.cleanup()
        assert analyzer._pipeline is None

    def test_temperature_scaling(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test temperature scaling for confidence calibration."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", temperature=0.5)
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("Test text")

            assert result.label == "positive"
            assert result.confidence != 0.90

    def test_max_length_truncation(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test that long texts are truncated to max_length."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", max_length=50, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            long_text = "This is a very long text. " * 100
            analyzer.analyze(long_text)

            call_args = mock_pipeline.call_args[0][0]
            assert len(call_args) <= 50


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_sentiment_analyzer(self, sentiment_module: Any) -> None:
        """Test get_sentiment_analyzer factory function."""
        get_sentiment_analyzer = sentiment_module.get_sentiment_analyzer
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        analyzer = get_sentiment_analyzer()

        assert isinstance(analyzer, FinBERTSentimentAnalyzer)
        assert analyzer.model_name == "ProsusAI/finbert"

    def test_get_sentiment_analyzer_with_config(self, sentiment_module: Any) -> None:
        """Test get_sentiment_analyzer with custom config."""
        SentimentConfig = sentiment_module.SentimentConfig
        get_sentiment_analyzer = sentiment_module.get_sentiment_analyzer
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        config = SentimentConfig(device="cpu", batch_size=32)
        analyzer = get_sentiment_analyzer(config)

        assert isinstance(analyzer, FinBERTSentimentAnalyzer)
        assert analyzer._config.batch_size == 32

    def test_analyze_financial_text(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test analyze_financial_text convenience function."""
        analyze_financial_text = sentiment_module.analyze_financial_text
        SentimentResult = sentiment_module.SentimentResult

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.92}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            result = analyze_financial_text("Strong performance this quarter.")

            assert isinstance(result, SentimentResult)
            assert result.label == "positive"
            assert result.confidence == 0.92

    def test_analyze_financial_text_empty(self, sentiment_module: Any) -> None:
        """Test analyze_financial_text with empty text."""
        analyze_financial_text = sentiment_module.analyze_financial_text
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyze_financial_text("")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_text(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test handling of very long text."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.80}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", max_length=512, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            very_long_text = "Financial market analysis. " * 1000
            result = analyzer.analyze(very_long_text)

            assert result.label == "positive"
            call_args = mock_pipeline.call_args[0][0]
            assert len(call_args) <= 512

    def test_special_characters(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test handling of special characters."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.70}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            text_with_special_chars = "Revenue: $1.5M (up 25%)"
            result = analyzer.analyze(text_with_special_chars)

            assert result.label == "neutral"

    def test_unicode_text(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test handling of Unicode text."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.85}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu")
            analyzer = FinBERTSentimentAnalyzer(config)

            unicode_text = "Les benefices ont augmente de 15%"
            result = analyzer.analyze(unicode_text)

            assert result.label == "positive"

    def test_single_word(self, sentiment_module: Any, mock_transformers: MagicMock) -> None:
        """Test handling of single word input."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.60}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu")
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("Growth")

            assert result.label == "positive"

    def test_text_becomes_empty_after_preprocessing(
        self, sentiment_module: Any, mock_transformers: MagicMock
    ) -> None:
        """Test handling when text becomes empty after preprocessing."""
        SentimentConfig = sentiment_module.SentimentConfig
        FinBERTSentimentAnalyzer = sentiment_module.FinBERTSentimentAnalyzer

        mock_pipeline = MagicMock()

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", preprocess_text=True)
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("http://example.com")

            assert result.label == "neutral"
            assert result.confidence == 1.0
