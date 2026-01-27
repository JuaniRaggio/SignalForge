"""Tests for sector classification module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from signalforge.nlp.embeddings import EmbeddingResult
from signalforge.nlp.sector_classifier import (
    GICS_SECTORS,
    BaseSectorClassifier,
    EmbeddingSectorClassifier,
    SectorClassifierConfig,
    SectorPrediction,
    classify_sector,
    get_all_sectors,
    get_sector_classifier,
)


class TestSectorPrediction:
    """Tests for SectorPrediction dataclass."""

    def test_valid_prediction(self) -> None:
        """Test creating a valid sector prediction."""
        all_scores = dict.fromkeys(GICS_SECTORS, 0.1)
        all_scores["Energy"] = 0.8

        prediction = SectorPrediction(
            text="Oil prices surge",
            sector="Energy",
            confidence=0.8,
            all_scores=all_scores,
        )

        assert prediction.text == "Oil prices surge"
        assert prediction.sector == "Energy"
        assert prediction.confidence == 0.8
        assert len(prediction.all_scores) == len(GICS_SECTORS)

    def test_invalid_sector(self) -> None:
        """Test that invalid sector raises ValueError."""
        all_scores = dict.fromkeys(GICS_SECTORS, 0.1)

        with pytest.raises(ValueError, match="Invalid sector"):
            SectorPrediction(
                text="Test",
                sector="InvalidSector",
                confidence=0.8,
                all_scores=all_scores,
            )

    def test_invalid_confidence_range(self) -> None:
        """Test that confidence outside 0-1 range raises ValueError."""
        all_scores = dict.fromkeys(GICS_SECTORS, 0.1)

        with pytest.raises(ValueError, match="Confidence must be between"):
            SectorPrediction(
                text="Test",
                sector="Energy",
                confidence=1.5,
                all_scores=all_scores,
            )

        with pytest.raises(ValueError, match="Confidence must be between"):
            SectorPrediction(
                text="Test",
                sector="Energy",
                confidence=-0.1,
                all_scores=all_scores,
            )

    def test_empty_scores(self) -> None:
        """Test that empty all_scores raises ValueError."""
        with pytest.raises(ValueError, match="all_scores dictionary cannot be empty"):
            SectorPrediction(
                text="Test",
                sector="Energy",
                confidence=0.8,
                all_scores={},
            )

    def test_invalid_score_in_dict(self) -> None:
        """Test that invalid score values raise ValueError."""
        all_scores = dict.fromkeys(GICS_SECTORS, 0.1)
        all_scores["Energy"] = 1.5

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            SectorPrediction(
                text="Test",
                sector="Financials",
                confidence=0.8,
                all_scores=all_scores,
            )

    def test_get_top_k_sectors(self) -> None:
        """Test getting top K sectors from prediction."""
        all_scores = {
            "Energy": 0.85,
            "Materials": 0.72,
            "Industrials": 0.68,
            "Consumer Discretionary": 0.45,
            "Consumer Staples": 0.42,
            "Health Care": 0.40,
            "Financials": 0.38,
            "Information Technology": 0.35,
            "Communication Services": 0.33,
            "Utilities": 0.30,
            "Real Estate": 0.28,
        }

        prediction = SectorPrediction(
            text="Test",
            sector="Energy",
            confidence=0.85,
            all_scores=all_scores,
        )

        top_3 = prediction.get_top_k_sectors(3)

        assert len(top_3) == 3
        assert top_3[0] == ("Energy", 0.85)
        assert top_3[1] == ("Materials", 0.72)
        assert top_3[2] == ("Industrials", 0.68)

    def test_get_top_k_more_than_available(self) -> None:
        """Test getting top K when K exceeds available sectors."""
        all_scores = dict.fromkeys(GICS_SECTORS, 0.1)
        all_scores["Energy"] = 0.8

        prediction = SectorPrediction(
            text="Test",
            sector="Energy",
            confidence=0.8,
            all_scores=all_scores,
        )

        top_k = prediction.get_top_k_sectors(20)

        assert len(top_k) == len(GICS_SECTORS)


class TestSectorClassifierConfig:
    """Tests for SectorClassifierConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SectorClassifierConfig()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.3
        assert config.top_k == 3
        assert config.normalize_embeddings is True
        assert config.cache_embeddings is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SectorClassifierConfig(
            model_name="all-mpnet-base-v2",
            similarity_threshold=0.5,
            top_k=5,
            normalize_embeddings=False,
            cache_embeddings=False,
        )

        assert config.model_name == "all-mpnet-base-v2"
        assert config.similarity_threshold == 0.5
        assert config.top_k == 5
        assert config.normalize_embeddings is False
        assert config.cache_embeddings is False

    def test_invalid_similarity_threshold(self) -> None:
        """Test that invalid similarity threshold raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            SectorClassifierConfig(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            SectorClassifierConfig(similarity_threshold=-0.1)

    def test_invalid_top_k(self) -> None:
        """Test that invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            SectorClassifierConfig(top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            SectorClassifierConfig(top_k=-1)

    def test_top_k_exceeds_sectors(self) -> None:
        """Test that top_k exceeding number of sectors is adjusted."""
        config = SectorClassifierConfig(top_k=50)

        assert config.top_k == len(GICS_SECTORS)


class TestBaseSectorClassifier:
    """Tests for BaseSectorClassifier abstract class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSectorClassifier()  # type: ignore


class TestEmbeddingSectorClassifier:
    """Tests for EmbeddingSectorClassifier."""

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedder for testing."""
        embedder = MagicMock()

        # Mock encode method
        def mock_encode(text: str) -> EmbeddingResult:
            # Return different embeddings based on text content
            if "oil" in text.lower() or "energy" in text.lower():
                embedding = [0.9, 0.1, 0.0] + [0.0] * 381  # Energy-like
            elif "bank" in text.lower() or "finance" in text.lower():
                embedding = [0.1, 0.9, 0.0] + [0.0] * 381  # Financials-like
            elif "tech" in text.lower() or "software" in text.lower():
                embedding = [0.0, 0.1, 0.9] + [0.0] * 381  # Tech-like
            else:
                embedding = [0.5, 0.5, 0.5] + [0.0] * 381  # Neutral

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name="all-MiniLM-L6-v2",
                dimension=384,
            )

        embedder.encode.side_effect = mock_encode

        # Mock encode_batch method
        def mock_encode_batch(texts: list[str]) -> list[EmbeddingResult]:
            return [mock_encode(text) for text in texts]

        embedder.encode_batch.side_effect = mock_encode_batch

        return embedder

    @pytest.fixture
    def classifier(self, mock_embedder: MagicMock) -> EmbeddingSectorClassifier:
        """Create a classifier with mocked embedder."""
        with patch("signalforge.nlp.sector_classifier.get_embedder") as mock_get:
            mock_get.return_value = mock_embedder
            classifier = EmbeddingSectorClassifier()
            # Pre-compute sector embeddings with mock
            classifier._ensure_sector_embeddings_cached()
            return classifier

    def test_initialization(self, mock_embedder: MagicMock) -> None:
        """Test classifier initialization."""
        with patch("signalforge.nlp.sector_classifier.get_embedder") as mock_get:
            mock_get.return_value = mock_embedder

            config = SectorClassifierConfig(similarity_threshold=0.5)
            classifier = EmbeddingSectorClassifier(config)

            assert classifier._config.similarity_threshold == 0.5
            assert classifier._embedder is not None

    def test_classify_empty_text(self, classifier: EmbeddingSectorClassifier) -> None:
        """Test that classifying empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            classifier.classify("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            classifier.classify("   ")

    def test_classify_single_text(self, classifier: EmbeddingSectorClassifier) -> None:
        """Test classifying a single text."""
        text = "Oil prices increased due to OPEC production cuts"
        prediction = classifier.classify(text)

        assert prediction.text == text
        assert prediction.sector in GICS_SECTORS
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.all_scores) == len(GICS_SECTORS)
        assert all(0.0 <= score <= 1.0 for score in prediction.all_scores.values())

    def test_classify_batch_empty_list(self, classifier: EmbeddingSectorClassifier) -> None:
        """Test that classifying empty list raises ValueError."""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            classifier.classify_batch([])

    def test_classify_batch_multiple_texts(
        self, classifier: EmbeddingSectorClassifier
    ) -> None:
        """Test batch classification of multiple texts."""
        texts = [
            "Banking sector profits increased",
            "Oil drilling operations expanded",
            "Software company launches AI product",
        ]

        predictions = classifier.classify_batch(texts)

        assert len(predictions) == len(texts)

        for i, prediction in enumerate(predictions):
            assert prediction.text == texts[i]
            assert prediction.sector in GICS_SECTORS
            assert 0.0 <= prediction.confidence <= 1.0
            assert len(prediction.all_scores) == len(GICS_SECTORS)

    def test_classify_batch_with_empty_text(
        self, classifier: EmbeddingSectorClassifier
    ) -> None:
        """Test batch classification with empty text in list."""
        texts = [
            "Banking profits rise",
            "",
            "Oil production increases",
        ]

        predictions = classifier.classify_batch(texts)

        assert len(predictions) == 3
        assert predictions[0].sector in GICS_SECTORS
        assert predictions[1].confidence == 0.0  # Empty text
        assert predictions[2].sector in GICS_SECTORS

    def test_sector_embeddings_cached(
        self, mock_embedder: MagicMock, classifier: EmbeddingSectorClassifier
    ) -> None:
        """Test that sector embeddings are cached."""
        # Embeddings should be computed during fixture setup
        assert len(classifier._sector_embeddings) == len(GICS_SECTORS)

        # Each sector should have multiple description embeddings
        for sector, embeddings in classifier._sector_embeddings.items():
            assert sector in GICS_SECTORS
            assert len(embeddings) > 0

    def test_low_confidence_warning(
        self, classifier: EmbeddingSectorClassifier, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that low confidence predictions generate warning."""
        # Create classifier with high threshold
        config = SectorClassifierConfig(similarity_threshold=0.99)
        with patch("signalforge.nlp.sector_classifier.get_embedder") as mock_get:
            mock_get.return_value = classifier._embedder
            high_threshold_classifier = EmbeddingSectorClassifier(config)
            high_threshold_classifier._sector_embeddings = classifier._sector_embeddings

        text = "Some ambiguous text"
        prediction = high_threshold_classifier.classify(text)

        # Check that prediction is made even with high threshold
        assert prediction.sector in GICS_SECTORS
        assert 0.0 <= prediction.confidence <= 1.0

        # If confidence is below threshold, warning should be logged
        if prediction.confidence < 0.99:
            # Verify warning was logged in this case
            assert any("low_confidence_prediction" in record.message for record in caplog.records)

    def test_metadata_in_prediction(self, classifier: EmbeddingSectorClassifier) -> None:
        """Test that prediction includes metadata."""
        text = "Banking sector analysis"
        prediction = classifier.classify(text)

        assert "model" in prediction.metadata
        assert "top_k_sectors" in prediction.metadata
        assert "threshold_met" in prediction.metadata

        top_k = prediction.metadata["top_k_sectors"]
        assert isinstance(top_k, dict)
        assert len(top_k) == classifier._config.top_k


class TestHelperFunctions:
    """Tests for module helper functions."""

    def test_get_all_sectors(self) -> None:
        """Test getting all GICS sectors."""
        sectors = get_all_sectors()

        assert len(sectors) == 11
        assert "Energy" in sectors
        assert "Materials" in sectors
        assert "Financials" in sectors
        assert "Information Technology" in sectors

        # Verify it returns a copy
        original_length = len(GICS_SECTORS)
        sectors.append("InvalidSector")
        assert len(GICS_SECTORS) == original_length

    @patch("signalforge.nlp.sector_classifier.get_embedder")
    def test_get_sector_classifier_default(self, mock_get_embedder: MagicMock) -> None:
        """Test getting default sector classifier."""
        mock_embedder = MagicMock()
        mock_get_embedder.return_value = mock_embedder

        # Reset global state
        import signalforge.nlp.sector_classifier as sc_module

        sc_module._default_classifier = None

        classifier1 = get_sector_classifier()
        classifier2 = get_sector_classifier()

        # Should return same instance
        assert classifier1 is classifier2

    @patch("signalforge.nlp.sector_classifier.get_embedder")
    def test_get_sector_classifier_custom(self, mock_get_embedder: MagicMock) -> None:
        """Test getting custom sector classifier."""
        mock_embedder = MagicMock()
        mock_get_embedder.return_value = mock_embedder

        config1 = SectorClassifierConfig(similarity_threshold=0.5)
        config2 = SectorClassifierConfig(similarity_threshold=0.6)

        classifier1 = get_sector_classifier(config1)
        classifier2 = get_sector_classifier(config2)

        # Should return different instances
        assert classifier1 is not classifier2
        assert classifier1._config.similarity_threshold == 0.5
        assert classifier2._config.similarity_threshold == 0.6

    @patch("signalforge.nlp.sector_classifier.get_sector_classifier")
    def test_classify_sector_convenience(self, mock_get_classifier: MagicMock) -> None:
        """Test convenience function for sector classification."""
        mock_classifier = MagicMock()
        mock_prediction = SectorPrediction(
            text="Test",
            sector="Energy",
            confidence=0.8,
            all_scores=dict.fromkeys(GICS_SECTORS, 0.1),
        )
        mock_classifier.classify.return_value = mock_prediction
        mock_get_classifier.return_value = mock_classifier

        text = "Oil prices surge"
        prediction = classify_sector(text)

        mock_get_classifier.assert_called_once_with()
        mock_classifier.classify.assert_called_once_with(text)
        assert prediction.sector == "Energy"


@pytest.mark.skip(reason="Integration tests require actual sentence-transformers models")
class TestIntegration:
    """Integration tests with actual embeddings (if available)."""

    @pytest.mark.integration
    def test_real_classification_energy(self) -> None:
        """Test real classification of energy-related text."""
        try:
            text = "Oil prices increased as OPEC announced production cuts"
            prediction = classify_sector(text)

            assert prediction.sector in GICS_SECTORS
            assert 0.0 <= prediction.confidence <= 1.0
            # Energy should likely be in top predictions
            top_3 = prediction.get_top_k_sectors(3)
            top_sectors = [s[0] for s in top_3]
            assert any(
                sector in ["Energy", "Materials", "Industrials"] for sector in top_sectors
            )

        except RuntimeError:
            pytest.skip("Sentence transformers not available")

    @pytest.mark.integration
    def test_real_classification_technology(self) -> None:
        """Test real classification of technology-related text."""
        try:
            text = "Apple announced new AI features in their software products"
            prediction = classify_sector(text)

            assert prediction.sector in GICS_SECTORS
            assert 0.0 <= prediction.confidence <= 1.0
            # Technology should likely be in top predictions
            top_3 = prediction.get_top_k_sectors(3)
            top_sectors = [s[0] for s in top_3]
            assert any(
                sector in ["Information Technology", "Communication Services"]
                for sector in top_sectors
            )

        except RuntimeError:
            pytest.skip("Sentence transformers not available")

    @pytest.mark.integration
    def test_real_batch_classification(self) -> None:
        """Test real batch classification."""
        try:
            classifier = get_sector_classifier()
            texts = [
                "Banking profits increased significantly",
                "New pharmaceutical drug approved by FDA",
                "Retail sales exceeded expectations",
            ]

            predictions = classifier.classify_batch(texts)

            assert len(predictions) == 3
            for prediction in predictions:
                assert prediction.sector in GICS_SECTORS
                assert 0.0 <= prediction.confidence <= 1.0

        except RuntimeError:
            pytest.skip("Sentence transformers not available")
