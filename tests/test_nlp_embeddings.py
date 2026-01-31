"""Tests for the NLP embeddings module.

This module tests the sentence-transformer based embedder with mocked models
to avoid requiring actual model downloads during testing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock, patch

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
        "sentence_transformers",
    ]

    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]

    # Install mocks
    sys.modules["torch"] = mock_torch
    sys.modules["torch.cuda"] = mock_torch.cuda
    sys.modules["torch.backends"] = mock_torch.backends
    sys.modules["torch.backends.mps"] = mock_torch.backends.mps

    mock_sentence_transformers = MagicMock()
    sys.modules["sentence_transformers"] = mock_sentence_transformers

    yield mock_torch

    # Restore original modules
    for mod_name in modules_to_mock:
        if mod_name in original_modules:
            sys.modules[mod_name] = original_modules[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]


@pytest.fixture
def embeddings_module(mock_torch_module: MagicMock) -> Any:
    """Import the embeddings module with mocked dependencies."""

    # Clear any cached imports of the embeddings module
    modules_to_clear = [k for k in list(sys.modules.keys()) if "signalforge.nlp.embeddings" in k]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    # Also clear the parent nlp module to force reimport
    if "signalforge.nlp" in sys.modules:
        del sys.modules["signalforge.nlp"]

    # Import fresh with mocked dependencies
    from signalforge.nlp import embeddings

    return embeddings


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_valid_embedding_result(self, embeddings_module: Any) -> None:
        """Test creating a valid embedding result."""
        EmbeddingResult = embeddings_module.EmbeddingResult
        embedding = [0.1, 0.2, 0.3, 0.4]
        result = EmbeddingResult(
            text="Test text",
            embedding=embedding,
            model_name="test-model",
            dimension=4,
        )

        assert result.text == "Test text"
        assert result.embedding == embedding
        assert result.model_name == "test-model"
        assert result.dimension == 4

    def test_dimension_mismatch_raises_error(self, embeddings_module: Any) -> None:
        """Test that dimension mismatch raises ValueError."""
        EmbeddingResult = embeddings_module.EmbeddingResult
        with pytest.raises(ValueError, match="does not match dimension"):
            EmbeddingResult(
                text="Test",
                embedding=[0.1, 0.2, 0.3],
                model_name="test-model",
                dimension=5,
            )

    def test_negative_dimension_raises_error(self, embeddings_module: Any) -> None:
        """Test that negative dimension raises ValueError."""
        EmbeddingResult = embeddings_module.EmbeddingResult
        with pytest.raises(ValueError, match="Dimension must be positive"):
            EmbeddingResult(
                text="Test",
                embedding=[],
                model_name="test-model",
                dimension=-1,
            )

    def test_nan_in_embedding_raises_error(self, embeddings_module: Any) -> None:
        """Test that NaN in embedding raises ValueError."""
        EmbeddingResult = embeddings_module.EmbeddingResult
        with pytest.raises(ValueError, match="invalid value"):
            EmbeddingResult(
                text="Test",
                embedding=[0.1, float("nan"), 0.3],
                model_name="test-model",
                dimension=3,
            )

    def test_inf_in_embedding_raises_error(self, embeddings_module: Any) -> None:
        """Test that infinity in embedding raises ValueError."""
        EmbeddingResult = embeddings_module.EmbeddingResult
        with pytest.raises(ValueError, match="invalid value"):
            EmbeddingResult(
                text="Test",
                embedding=[0.1, float("inf"), 0.3],
                model_name="test-model",
                dimension=3,
            )

    def test_non_numeric_embedding_value_raises_error(self, embeddings_module: Any) -> None:
        """Test that non-numeric values in embedding raise ValueError."""
        EmbeddingResult = embeddings_module.EmbeddingResult
        with pytest.raises(ValueError, match="not numeric"):
            EmbeddingResult(
                text="Test",
                embedding=[0.1, "invalid", 0.3],  # type: ignore[list-item]
                model_name="test-model",
                dimension=3,
            )


class TestEmbeddingsConfig:
    """Tests for EmbeddingsConfig dataclass."""

    def test_default_config(self, embeddings_module: Any) -> None:
        """Test default configuration values."""
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        config = EmbeddingsConfig()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "auto"
        assert config.normalize is True
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.cache_model is True

    def test_custom_config(self, embeddings_module: Any) -> None:
        """Test custom configuration."""
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        config = EmbeddingsConfig(
            model_name="all-mpnet-base-v2",
            device="cpu",
            normalize=False,
            batch_size=16,
            max_length=256,
            cache_model=False,
        )

        assert config.model_name == "all-mpnet-base-v2"
        assert config.device == "cpu"
        assert config.normalize is False
        assert config.batch_size == 16
        assert config.max_length == 256
        assert config.cache_model is False

    def test_invalid_batch_size_raises_error(self, embeddings_module: Any) -> None:
        """Test that invalid batch_size raises ValueError."""
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingsConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingsConfig(batch_size=-1)

    def test_invalid_max_length_raises_error(self, embeddings_module: Any) -> None:
        """Test that invalid max_length raises ValueError."""
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        with pytest.raises(ValueError, match="max_length must be positive"):
            EmbeddingsConfig(max_length=0)

        with pytest.raises(ValueError, match="max_length must be positive"):
            EmbeddingsConfig(max_length=-10)

    def test_invalid_device_raises_error(self, embeddings_module: Any) -> None:
        """Test that invalid device raises ValueError."""
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        with pytest.raises(ValueError, match="device must be one of"):
            EmbeddingsConfig(device="invalid")


class TestBaseEmbeddingModel:
    """Tests for BaseEmbeddingModel abstract class."""

    def test_base_class_is_abstract(self, embeddings_module: Any) -> None:
        """Test that BaseEmbeddingModel cannot be instantiated."""
        BaseEmbeddingModel = embeddings_module.BaseEmbeddingModel
        with pytest.raises(TypeError):
            BaseEmbeddingModel()  # type: ignore[abstract]


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder class."""

    @pytest.fixture(autouse=True)
    def clear_model_cache(self, embeddings_module: Any) -> None:
        """Clear model cache before each test."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        SentenceTransformerEmbedder._model_cache.clear()

    @pytest.fixture
    def mock_sentence_transformer(self) -> MagicMock:
        """Create a mock SentenceTransformer model."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512
        return mock_model

    @pytest.fixture
    def mock_torch_no_gpu(self) -> Mock:
        """Mock torch with no GPU available."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        return mock_torch

    def test_embedder_initialization(self, embeddings_module: Any) -> None:
        """Test embedder initialization with default config."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder._config.device == "auto"
        assert embedder._config.normalize is True

    def test_embedder_with_custom_config(self, embeddings_module: Any) -> None:
        """Test embedder initialization with custom config."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        config = EmbeddingsConfig(
            model_name="all-mpnet-base-v2",
            device="cpu",
            normalize=False,
        )
        embedder = SentenceTransformerEmbedder(config)

        assert embedder.model_name == "all-mpnet-base-v2"
        assert embedder._config.device == "cpu"
        assert embedder._config.normalize is False

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_device_auto_no_gpu(
        self,
        mock_st_class: Mock,
        mock_torch_no_gpu: Mock,
        embeddings_module: Any,
    ) -> None:
        """Test device selection with auto and no GPU."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        with patch.dict("sys.modules", {"torch": mock_torch_no_gpu}):
            embedder = SentenceTransformerEmbedder(EmbeddingsConfig(device="auto"))
            device = embedder._get_device()

            assert device == "cpu"

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_device_cpu(self, mock_st_class: Mock, embeddings_module: Any) -> None:
        """Test device selection with explicit CPU."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig
        embedder = SentenceTransformerEmbedder(EmbeddingsConfig(device="cpu"))
        device = embedder._get_device()

        assert device == "cpu"

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_single_text(
        self,
        mock_st_class: Mock,
        mock_sentence_transformer: MagicMock,
        embeddings_module: Any,
    ) -> None:
        """Test encoding a single text."""
        import numpy as np

        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder

        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_sentence_transformer.encode.return_value = mock_embedding
        mock_st_class.return_value = mock_sentence_transformer

        embedder = SentenceTransformerEmbedder()
        result = embedder.encode("Test text")

        assert result.text == "Test text"
        assert result.embedding == [0.1, 0.2, 0.3, 0.4]
        assert result.model_name == "all-MiniLM-L6-v2"
        assert result.dimension == 4

        mock_sentence_transformer.encode.assert_called_once()
        call_args = mock_sentence_transformer.encode.call_args
        assert call_args[0][0] == "Test text"
        assert call_args[1]["normalize_embeddings"] is True
        assert call_args[1]["convert_to_numpy"] is True

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_empty_text_raises_error(
        self, mock_st_class: Mock, embeddings_module: Any
    ) -> None:
        """Test that encoding empty text raises ValueError."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedder.encode("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            embedder.encode("   ")

    def test_encode_batch(
        self,
        mock_sentence_transformer: MagicMock,
        embeddings_module: Any,
    ) -> None:
        """Test encoding multiple texts in batch."""
        import numpy as np

        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder

        mock_embeddings = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
            ]
        )
        mock_sentence_transformer.encode.return_value = mock_embeddings

        embedder = SentenceTransformerEmbedder()
        embedder._model = mock_sentence_transformer

        texts = ["Text 1", "Text 2", "Text 3"]
        results = embedder.encode_batch(texts)

        assert len(results) == 3
        assert results[0].text == "Text 1"
        assert list(results[0].embedding) == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert results[1].text == "Text 2"
        assert list(results[1].embedding) == pytest.approx([0.5, 0.6, 0.7, 0.8])
        assert results[2].text == "Text 3"
        assert list(results[2].embedding) == pytest.approx([0.9, 1.0, 1.1, 1.2])

        mock_sentence_transformer.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_batch_empty_list_raises_error(
        self, mock_st_class: Mock, embeddings_module: Any
    ) -> None:
        """Test that encoding empty list raises ValueError."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            embedder.encode_batch([])

    def test_encode_batch_with_empty_texts(
        self,
        mock_sentence_transformer: MagicMock,
        embeddings_module: Any,
    ) -> None:
        """Test batch encoding with some empty texts."""
        import numpy as np

        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder

        mock_sentence_transformer.get_sentence_embedding_dimension.return_value = 4
        mock_embeddings = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
        )
        mock_sentence_transformer.encode.return_value = mock_embeddings

        embedder = SentenceTransformerEmbedder()
        embedder._model = mock_sentence_transformer

        texts = ["Text 1", "", "Text 2", "   "]
        results = embedder.encode_batch(texts)

        assert len(results) == 4
        assert results[0].text == "Text 1"
        assert list(results[0].embedding) == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert results[1].text == ""
        assert results[1].embedding == [0.0, 0.0, 0.0, 0.0]
        assert results[2].text == "Text 2"
        assert list(results[2].embedding) == pytest.approx([0.5, 0.6, 0.7, 0.8])
        assert results[3].text == "   "
        assert results[3].embedding == [0.0, 0.0, 0.0, 0.0]

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_batch_all_empty_raises_error(
        self,
        mock_st_class: Mock,
        mock_sentence_transformer: MagicMock,
        embeddings_module: Any,
    ) -> None:
        """Test that encoding all empty texts raises ValueError."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        mock_st_class.return_value = mock_sentence_transformer

        embedder = SentenceTransformerEmbedder()

        with pytest.raises(ValueError, match="All texts are empty"):
            embedder.encode_batch(["", "   ", "\n\t"])

    @patch("sentence_transformers.SentenceTransformer")
    def test_dimension_property(
        self,
        mock_st_class: Mock,
        mock_sentence_transformer: MagicMock,
        embeddings_module: Any,
    ) -> None:
        """Test dimension property."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        mock_st_class.return_value = mock_sentence_transformer

        embedder = SentenceTransformerEmbedder()
        assert embedder.dimension == 384
        assert embedder.dimension == 384
        assert embedder._dimension == 384

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_caching(
        self,
        mock_st_class: Mock,
        mock_sentence_transformer: MagicMock,
        embeddings_module: Any,
    ) -> None:
        """Test that models are cached when cache_model=True."""
        import numpy as np

        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig

        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_sentence_transformer.encode.return_value = mock_embedding
        mock_st_class.return_value = mock_sentence_transformer

        SentenceTransformerEmbedder._model_cache.clear()

        config = EmbeddingsConfig(cache_model=True)
        embedder1 = SentenceTransformerEmbedder(config)
        embedder1.encode("Test")

        embedder2 = SentenceTransformerEmbedder(config)
        embedder2.encode("Test 2")

        assert mock_st_class.call_count == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_cleanup(
        self,
        mock_st_class: Mock,
        mock_sentence_transformer: MagicMock,
        mock_torch_no_gpu: Mock,
        embeddings_module: Any,
    ) -> None:
        """Test cleanup method."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        with patch.dict("sys.modules", {"torch": mock_torch_no_gpu}):
            mock_st_class.return_value = mock_sentence_transformer

            embedder = SentenceTransformerEmbedder()
            embedder._model = mock_sentence_transformer
            embedder._dimension = 384

            embedder.cleanup()

            assert embedder._model is None
            assert embedder._dimension is None

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loading_error(self, mock_st_class: Mock, embeddings_module: Any) -> None:
        """Test error handling during model loading."""
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        mock_st_class.side_effect = Exception("Model loading failed")

        embedder = SentenceTransformerEmbedder()

        with pytest.raises(RuntimeError, match="Failed to load model"):
            embedder.encode("Test text")


class TestHelperFunctions:
    """Tests for helper functions."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embedder_default(self, mock_st_class: Mock, embeddings_module: Any) -> None:
        """Test get_embedder with default config."""
        get_embedder = embeddings_module.get_embedder
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder

        embeddings_module._default_embedder = None

        embedder = get_embedder()

        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder.model_name == "all-MiniLM-L6-v2"

        embedder2 = get_embedder()
        assert embedder is embedder2

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embedder_custom_config(self, mock_st_class: Mock, embeddings_module: Any) -> None:
        """Test get_embedder with custom config."""
        get_embedder = embeddings_module.get_embedder
        SentenceTransformerEmbedder = embeddings_module.SentenceTransformerEmbedder
        EmbeddingsConfig = embeddings_module.EmbeddingsConfig

        config = EmbeddingsConfig(model_name="all-mpnet-base-v2")
        embedder = get_embedder(config)

        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder.model_name == "all-mpnet-base-v2"

    def test_embed_text(self, embeddings_module: Any) -> None:
        """Test embed_text convenience function."""
        embed_text = embeddings_module.embed_text
        EmbeddingResult = embeddings_module.EmbeddingResult

        with patch.object(embeddings_module, "get_embedder") as mock_get_embedder:
            mock_embedder = Mock()
            mock_result = EmbeddingResult(
                text="Test",
                embedding=[0.1, 0.2, 0.3],
                model_name="test-model",
                dimension=3,
            )
            mock_embedder.encode.return_value = mock_result
            mock_get_embedder.return_value = mock_embedder

            result = embed_text("Test text")

            assert result == mock_result
            mock_embedder.encode.assert_called_once_with("Test text")

    def test_embed_texts(self, embeddings_module: Any) -> None:
        """Test embed_texts convenience function."""
        embed_texts = embeddings_module.embed_texts
        EmbeddingResult = embeddings_module.EmbeddingResult

        with patch.object(embeddings_module, "get_embedder") as mock_get_embedder:
            mock_embedder = Mock()
            mock_results = [
                EmbeddingResult(
                    text="Test 1",
                    embedding=[0.1, 0.2, 0.3],
                    model_name="test-model",
                    dimension=3,
                ),
                EmbeddingResult(
                    text="Test 2",
                    embedding=[0.4, 0.5, 0.6],
                    model_name="test-model",
                    dimension=3,
                ),
            ]
            mock_embedder.encode_batch.return_value = mock_results
            mock_get_embedder.return_value = mock_embedder

            texts = ["Test 1", "Test 2"]
            results = embed_texts(texts)

            assert results == mock_results
            mock_embedder.encode_batch.assert_called_once_with(texts)


class TestComputeSimilarity:
    """Tests for compute_similarity function."""

    def test_identical_vectors(self, embeddings_module: Any) -> None:
        """Test similarity of identical vectors."""
        compute_similarity = embeddings_module.compute_similarity
        emb = [0.5, 0.5, 0.0]
        similarity = compute_similarity(emb, emb)

        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self, embeddings_module: Any) -> None:
        """Test similarity of orthogonal vectors."""
        compute_similarity = embeddings_module.compute_similarity
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity = compute_similarity(emb1, emb2)

        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self, embeddings_module: Any) -> None:
        """Test similarity of opposite vectors."""
        compute_similarity = embeddings_module.compute_similarity
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [-1.0, 0.0, 0.0]
        similarity = compute_similarity(emb1, emb2)

        assert similarity == pytest.approx(-1.0, abs=1e-6)

    def test_normalized_vectors(self, embeddings_module: Any) -> None:
        """Test similarity of normalized vectors (typical use case)."""
        compute_similarity = embeddings_module.compute_similarity
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]

        similarity = compute_similarity(emb1, emb2)

        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_zero(self, embeddings_module: Any) -> None:
        """Test similarity with zero vector."""
        compute_similarity = embeddings_module.compute_similarity
        emb1 = [0.0, 0.0, 0.0]
        emb2 = [1.0, 2.0, 3.0]

        similarity = compute_similarity(emb1, emb2)
        assert similarity == 0.0

        similarity = compute_similarity(emb2, emb1)
        assert similarity == 0.0

    def test_empty_embeddings_raises_error(self, embeddings_module: Any) -> None:
        """Test that empty embeddings raise ValueError."""
        compute_similarity = embeddings_module.compute_similarity
        with pytest.raises(ValueError, match="Embeddings cannot be empty"):
            compute_similarity([], [1.0, 2.0])

        with pytest.raises(ValueError, match="Embeddings cannot be empty"):
            compute_similarity([1.0, 2.0], [])

    def test_mismatched_dimensions_raises_error(self, embeddings_module: Any) -> None:
        """Test that mismatched dimensions raise ValueError."""
        compute_similarity = embeddings_module.compute_similarity
        emb1 = [1.0, 2.0, 3.0]
        emb2 = [1.0, 2.0]

        with pytest.raises(ValueError, match="must have same dimension"):
            compute_similarity(emb1, emb2)

    def test_similarity_range_clamping(self, embeddings_module: Any) -> None:
        """Test that similarity is clamped to [-1, 1] range."""
        compute_similarity = embeddings_module.compute_similarity
        emb1 = [1.0, 0.0]
        emb2 = [1.0000001, 0.0]

        similarity = compute_similarity(emb1, emb2)

        assert -1.0 <= similarity <= 1.0
