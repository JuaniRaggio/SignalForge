"""Tests for sentiment analysis module.

This module tests the FinBERT-based sentiment analyzer with mocked models
to avoid requiring actual model downloads during testing.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock torch and transformers modules before importing sentiment module
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
mock_torch.cuda.empty_cache = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_torch.cuda
sys.modules["torch.backends"] = mock_torch.backends
sys.modules["torch.backends.mps"] = mock_torch.backends.mps

mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers

# ruff: noqa: E402
from signalforge.nlp.sentiment import (
    BaseSentimentAnalyzer,
    FinBERTSentimentAnalyzer,
    SentimentConfig,
    SentimentResult,
    analyze_financial_text,
    get_sentiment_analyzer,
)


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_valid_sentiment_result(self) -> None:
        """Test creation of valid sentiment result."""
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

    def test_invalid_confidence_range(self) -> None:
        """Test that invalid confidence values raise ValueError."""
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

    def test_invalid_label(self) -> None:
        """Test that invalid labels raise ValueError."""
        with pytest.raises(ValueError, match="Invalid label"):
            SentimentResult(
                text="Test",
                label="invalid",  # type: ignore[arg-type]
                confidence=0.8,
                scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1},
            )

    def test_invalid_scores_keys(self) -> None:
        """Test that missing or extra score keys raise ValueError."""
        with pytest.raises(ValueError, match="Scores must contain exactly"):
            SentimentResult(
                text="Test",
                label="positive",
                confidence=0.8,
                scores={"positive": 0.8, "negative": 0.2},  # Missing neutral
            )

    def test_invalid_score_values(self) -> None:
        """Test that score values outside 0-1 range raise ValueError."""
        with pytest.raises(ValueError, match="Score for .* must be between 0.0 and 1.0"):
            SentimentResult(
                text="Test",
                label="positive",
                confidence=0.8,
                scores={"positive": 1.5, "negative": 0.1, "neutral": 0.1},
            )


class TestSentimentConfig:
    """Tests for SentimentConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SentimentConfig()

        assert config.model_name == "ProsusAI/finbert"
        assert config.device == "auto"
        assert config.batch_size == 16
        assert config.max_length == 512
        assert config.cache_model is True
        assert config.preprocess_text is False
        assert config.temperature == 1.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
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

    def test_invalid_batch_size(self) -> None:
        """Test that invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SentimentConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            SentimentConfig(batch_size=-1)

    def test_invalid_max_length(self) -> None:
        """Test that invalid max length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            SentimentConfig(max_length=0)

        with pytest.raises(ValueError, match="max_length must be positive"):
            SentimentConfig(max_length=-1)

    def test_invalid_device(self) -> None:
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            SentimentConfig(device="invalid")

    def test_invalid_temperature(self) -> None:
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            SentimentConfig(temperature=0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            SentimentConfig(temperature=-1)


class TestBaseSentimentAnalyzer:
    """Tests for BaseSentimentAnalyzer abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSentimentAnalyzer()  # type: ignore[abstract]


class TestFinBERTSentimentAnalyzer:
    """Tests for FinBERTSentimentAnalyzer class."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        config = SentimentConfig(device="cpu")
        analyzer = FinBERTSentimentAnalyzer(config)

        assert analyzer.model_name == "ProsusAI/finbert"
        assert analyzer._config.device == "cpu"
        assert analyzer._pipeline is None  # Lazy loading

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        config = SentimentConfig(model_name="test/model")
        analyzer = FinBERTSentimentAnalyzer(config)

        assert analyzer.model_name == "test/model"

    def test_get_device_cpu(self) -> None:
        """Test device selection for CPU."""
        config = SentimentConfig(device="cpu")
        analyzer = FinBERTSentimentAnalyzer(config)

        device = analyzer._get_device()
        assert device == -1

    def test_get_device_cuda_available(self) -> None:
        """Test device selection when CUDA is available."""
        with patch.object(mock_torch.cuda, "is_available", return_value=True):
            config = SentimentConfig(device="auto")
            analyzer = FinBERTSentimentAnalyzer(config)

            device = analyzer._get_device()
            assert device == 0

    def test_get_device_auto_fallback_cpu(self) -> None:
        """Test device selection falls back to CPU when GPU unavailable."""
        with patch.object(mock_torch.cuda, "is_available", return_value=False), patch.object(
            mock_torch.backends.mps, "is_available", return_value=False
        ):
            config = SentimentConfig(device="auto")
            analyzer = FinBERTSentimentAnalyzer(config)

            device = analyzer._get_device()
            assert device == -1

    def test_analyze_empty_text(self) -> None:
        """Test that analyzing empty text raises ValueError."""
        analyzer = FinBERTSentimentAnalyzer()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.analyze("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyzer.analyze("   ")

    def test_analyze_single_text(self) -> None:
        """Test analyzing single text."""
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

    def test_analyze_negative_sentiment(self) -> None:
        """Test analyzing text with negative sentiment."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "negative", "score": 0.88}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("Company missed earnings expectations.")

            assert result.label == "negative"
            assert result.confidence == 0.88

    def test_analyze_neutral_sentiment(self) -> None:
        """Test analyzing text with neutral sentiment."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.75}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("The company reported quarterly results.")

            assert result.label == "neutral"
            assert result.confidence == 0.75

    def test_analyze_batch_empty_list(self) -> None:
        """Test that analyzing empty batch raises ValueError."""
        analyzer = FinBERTSentimentAnalyzer()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            analyzer.analyze_batch([])

    def test_analyze_batch(self) -> None:
        """Test batch sentiment analysis."""
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

    def test_analyze_batch_with_empty_texts(self) -> None:
        """Test batch analysis with some empty texts."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu")
            analyzer = FinBERTSentimentAnalyzer(config)

            texts = ["Good earnings.", "", "   "]
            results = analyzer.analyze_batch(texts)

            assert len(results) == 3
            assert results[0].label == "positive"
            # Empty texts should return neutral
            assert results[1].label == "neutral"
            assert results[2].label == "neutral"

    def test_analyze_batch_multiple_batches(self) -> None:
        """Test batch analysis with multiple batches."""
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
            assert mock_pipeline.call_count == 2  # Two batches

    def test_analyze_with_preprocessing(self) -> None:
        """Test analysis with text preprocessing enabled."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", preprocess_text=True, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            # Text with URLs and extra whitespace
            text = "  Strong earnings!  http://example.com  "
            result = analyzer.analyze(text)

            assert result.label == "positive"
            # Verify preprocessing was called (URL should be removed)
            call_args = mock_pipeline.call_args[0][0]
            assert "http://example.com" not in call_args

    def test_model_caching(self) -> None:
        """Test that model is loaded only once when caching is enabled."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline) as mock_factory:
            config = SentimentConfig(device="cpu", cache_model=True)
            analyzer = FinBERTSentimentAnalyzer(config)

            # Clear cache before test
            analyzer._model_cache.clear()

            # First analysis should load model
            analyzer.analyze("Text 1")
            first_call_count = mock_factory.call_count

            # Second analysis should use cached model
            analyzer.analyze("Text 2")
            second_call_count = mock_factory.call_count

            assert second_call_count == first_call_count  # Model loaded only once

    def test_model_load_failure_import_error(self) -> None:
        """Test handling of model loading failure due to missing dependencies."""
        with patch.object(
            mock_transformers, "pipeline", side_effect=ImportError("transformers not installed")
        ):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            with pytest.raises(RuntimeError, match="transformers or torch not installed"):
                analyzer.analyze("Test text")

    def test_model_load_failure_general_error(self) -> None:
        """Test handling of general model loading failure."""
        with patch.object(
            mock_transformers, "pipeline", side_effect=Exception("Model download failed")
        ):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            with pytest.raises(RuntimeError, match="Failed to load model"):
                analyzer.analyze("Test text")

    def test_cleanup_gpu_memory(self) -> None:
        """Test GPU memory cleanup."""
        with patch.object(mock_torch.cuda, "is_available", return_value=True):
            analyzer = FinBERTSentimentAnalyzer()
            analyzer._pipeline = MagicMock()  # Simulate loaded model

            analyzer.cleanup()

            mock_torch.cuda.empty_cache.assert_called_once()
            assert analyzer._pipeline is None

    def test_cleanup_no_gpu(self) -> None:
        """Test cleanup when no GPU is used."""
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock()

        # Should not raise exception
        analyzer.cleanup()
        assert analyzer._pipeline is None

    def test_temperature_scaling(self) -> None:
        """Test temperature scaling for confidence calibration."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            # Test with temperature scaling
            config = SentimentConfig(device="cpu", temperature=0.5)
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("Test text")

            # Temperature scaling should affect confidence
            # With temperature < 1, high confidence should become even higher
            assert result.label == "positive"
            assert result.confidence != 0.90  # Should be scaled

    def test_max_length_truncation(self) -> None:
        """Test that long texts are truncated to max_length."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.90}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", max_length=50, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            # Create a long text
            long_text = "This is a very long text. " * 100
            analyzer.analyze(long_text)

            # Verify pipeline was called with truncated text
            call_args = mock_pipeline.call_args[0][0]
            assert len(call_args) <= 50


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_sentiment_analyzer(self) -> None:
        """Test get_sentiment_analyzer factory function."""
        analyzer = get_sentiment_analyzer()

        assert isinstance(analyzer, FinBERTSentimentAnalyzer)
        assert analyzer.model_name == "ProsusAI/finbert"

    def test_get_sentiment_analyzer_with_config(self) -> None:
        """Test get_sentiment_analyzer with custom config."""
        config = SentimentConfig(device="cpu", batch_size=32)
        analyzer = get_sentiment_analyzer(config)

        assert isinstance(analyzer, FinBERTSentimentAnalyzer)
        assert analyzer._config.batch_size == 32

    def test_analyze_financial_text(self) -> None:
        """Test analyze_financial_text convenience function."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.92}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            result = analyze_financial_text("Strong performance this quarter.")

            assert isinstance(result, SentimentResult)
            assert result.label == "positive"
            assert result.confidence == 0.92

    def test_analyze_financial_text_empty(self) -> None:
        """Test analyze_financial_text with empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyze_financial_text("")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_text(self) -> None:
        """Test handling of very long text."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.80}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", max_length=512, cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            # Create very long text
            very_long_text = "Financial market analysis. " * 1000
            result = analyzer.analyze(very_long_text)

            assert result.label == "positive"
            # Text should be truncated
            call_args = mock_pipeline.call_args[0][0]
            assert len(call_args) <= 512

    def test_special_characters(self) -> None:
        """Test handling of special characters."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.70}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", cache_model=False)
            analyzer = FinBERTSentimentAnalyzer(config)

            text_with_special_chars = "Revenue: $1.5M (up 25%)"
            result = analyzer.analyze(text_with_special_chars)

            assert result.label == "neutral"

    def test_unicode_text(self) -> None:
        """Test handling of Unicode text."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.85}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu")
            analyzer = FinBERTSentimentAnalyzer(config)

            unicode_text = "Les bénéfices ont augmenté de 15%"
            result = analyzer.analyze(unicode_text)

            assert result.label == "positive"

    def test_single_word(self) -> None:
        """Test handling of single word input."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.60}]

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu")
            analyzer = FinBERTSentimentAnalyzer(config)

            result = analyzer.analyze("Growth")

            assert result.label == "positive"

    def test_text_becomes_empty_after_preprocessing(self) -> None:
        """Test handling when text becomes empty after preprocessing."""
        mock_pipeline = MagicMock()

        with patch.object(mock_transformers, "pipeline", return_value=mock_pipeline):
            config = SentimentConfig(device="cpu", preprocess_text=True)
            analyzer = FinBERTSentimentAnalyzer(config)

            # Text that might become empty after preprocessing (just a URL)
            result = analyzer.analyze("http://example.com")

            # Should return neutral for empty preprocessed text
            assert result.label == "neutral"
            assert result.confidence == 1.0
