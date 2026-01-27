"""Natural Language Processing module for financial text analysis.

This module provides text preprocessing, sentiment analysis, named entity
recognition, and embedding generation capabilities specifically designed
for financial documents.
"""

from signalforge.nlp.embeddings import (
    BaseEmbeddingModel,
    EmbeddingResult,
    EmbeddingsConfig,
    SentenceTransformerEmbedder,
    compute_similarity,
    embed_text,
    embed_texts,
    get_embedder,
)
from signalforge.nlp.ner import (
    BaseEntityExtractor,
    EntityExtractionResult,
    FinancialEntityExtractor,
    NERConfig,
    NamedEntity,
    SpaCyEntityExtractor,
    extract_entities,
    extract_tickers,
    get_entity_extractor,
)
from signalforge.nlp.preprocessing import (
    DocumentPreprocessor,
    PreprocessingConfig,
    ProcessedDocument,
    TextPreprocessor,
)
from signalforge.nlp.sentiment import (
    BaseSentimentAnalyzer,
    FinBERTSentimentAnalyzer,
    SentimentConfig,
    SentimentResult,
    analyze_financial_text,
    get_sentiment_analyzer,
)

__all__ = [
    # Embeddings
    "BaseEmbeddingModel",
    "EmbeddingResult",
    "EmbeddingsConfig",
    "SentenceTransformerEmbedder",
    "compute_similarity",
    "embed_text",
    "embed_texts",
    "get_embedder",
    # Named Entity Recognition
    "BaseEntityExtractor",
    "EntityExtractionResult",
    "FinancialEntityExtractor",
    "NERConfig",
    "NamedEntity",
    "SpaCyEntityExtractor",
    "extract_entities",
    "extract_tickers",
    "get_entity_extractor",
    # Preprocessing
    "DocumentPreprocessor",
    "PreprocessingConfig",
    "ProcessedDocument",
    "TextPreprocessor",
    # Sentiment Analysis
    "BaseSentimentAnalyzer",
    "FinBERTSentimentAnalyzer",
    "SentimentConfig",
    "SentimentResult",
    "analyze_financial_text",
    "get_sentiment_analyzer",
]
