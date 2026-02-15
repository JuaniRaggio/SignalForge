"""Natural Language Processing module for financial text analysis.

This module provides text preprocessing, sentiment analysis, named entity
recognition, embedding generation, and vector storage capabilities specifically
designed for financial documents.
"""

from signalforge.nlp.contradiction_detector import (
    AnalystOpinion,
    Contradiction,
    ContradictionDetector,
    ContradictionType,
    DivergenceAnalysis,
    DivergenceSeverity,
)
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
    NamedEntity,
    NERConfig,
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
from signalforge.nlp.sector_classifier import (
    BaseSectorClassifier,
    EmbeddingSectorClassifier,
    SectorClassifierConfig,
    SectorPrediction,
    classify_sector,
    get_all_sectors,
    get_sector_classifier,
)
from signalforge.nlp.sentiment import (
    BaseSentimentAnalyzer,
    FinBERTSentimentAnalyzer,
    SentimentConfig,
    SentimentResult,
    analyze_financial_text,
    get_sentiment_analyzer,
)
from signalforge.nlp.summarization import (
    MultiDocumentSummary,
    SummaryLength,
    SummaryResult,
    SummaryStyle,
    TextSummarizer,
)
from signalforge.nlp.topics import (
    BaseTopicExtractor,
    EmbeddingTopicExtractor,
    TopicExtractionConfig,
    TopicExtractionResult,
    TopicKeyword,
    extract_keyphrases,
    extract_topics,
    get_topic_extractor,
)
from signalforge.nlp.vector_store import (
    VectorSearchResult,
    VectorStore,
    VectorStoreConfig,
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
    # Sector Classification
    "BaseSectorClassifier",
    "EmbeddingSectorClassifier",
    "SectorClassifierConfig",
    "SectorPrediction",
    "classify_sector",
    "get_all_sectors",
    "get_sector_classifier",
    # Sentiment Analysis
    "BaseSentimentAnalyzer",
    "FinBERTSentimentAnalyzer",
    "SentimentConfig",
    "SentimentResult",
    "analyze_financial_text",
    "get_sentiment_analyzer",
    # Topic Extraction
    "BaseTopicExtractor",
    "EmbeddingTopicExtractor",
    "TopicExtractionConfig",
    "TopicExtractionResult",
    "TopicKeyword",
    "extract_keyphrases",
    "extract_topics",
    "get_topic_extractor",
    # Vector Store
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreConfig",
    # Contradiction Detection
    "AnalystOpinion",
    "Contradiction",
    "ContradictionDetector",
    "ContradictionType",
    "DivergenceAnalysis",
    "DivergenceSeverity",
    # Summarization
    "MultiDocumentSummary",
    "SummaryLength",
    "SummaryResult",
    "SummaryStyle",
    "TextSummarizer",
]
