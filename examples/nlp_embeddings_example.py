"""Example usage of the embeddings module.

This script demonstrates how to use the sentence-transformer embeddings
module for financial text analysis.
"""

from signalforge.nlp.embeddings import (
    EmbeddingsConfig,
    compute_similarity,
    embed_text,
    embed_texts,
    get_embedder,
)


def example_basic_embedding() -> None:
    """Demonstrate basic text embedding."""
    print("=" * 80)
    print("Example 1: Basic Text Embedding")
    print("=" * 80)

    text = "Apple Inc. reported record quarterly revenue of $123.9 billion."

    # Generate embedding using convenience function
    result = embed_text(text)

    print(f"Text: {result.text}")
    print(f"Model: {result.model_name}")
    print(f"Dimension: {result.dimension}")
    print(f"Embedding (first 5 values): {result.embedding[:5]}")
    print()


def example_batch_embedding() -> None:
    """Demonstrate batch text embedding."""
    print("=" * 80)
    print("Example 2: Batch Text Embedding")
    print("=" * 80)

    texts = [
        "Tesla stock surged 10% on strong delivery numbers.",
        "Amazon reported disappointing earnings in Q3.",
        "Microsoft announces new AI features for Office 365.",
        "Meta faces regulatory scrutiny in Europe.",
    ]

    # Generate embeddings in batch
    results = embed_texts(texts)

    for idx, result in enumerate(results, 1):
        print(f"{idx}. Text: {result.text[:50]}...")
        print(f"   Dimension: {result.dimension}")
        print(f"   Embedding (first 3): {result.embedding[:3]}")
        print()


def example_similarity_computation() -> None:
    """Demonstrate similarity computation between texts."""
    print("=" * 80)
    print("Example 3: Computing Text Similarity")
    print("=" * 80)

    # Financial news about the same topic
    text1 = "Apple stock rose 5% after earnings beat expectations."
    text2 = "AAPL shares jumped following strong quarterly results."
    text3 = "Oil prices declined amid concerns about global demand."

    # Generate embeddings
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    emb3 = embed_text(text3)

    # Compute similarities
    sim_1_2 = compute_similarity(emb1.embedding, emb2.embedding)
    sim_1_3 = compute_similarity(emb1.embedding, emb3.embedding)
    sim_2_3 = compute_similarity(emb2.embedding, emb3.embedding)

    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    print()
    print(f"Similarity (Text 1 vs Text 2): {sim_1_2:.4f}")
    print(f"Similarity (Text 1 vs Text 3): {sim_1_3:.4f}")
    print(f"Similarity (Text 2 vs Text 3): {sim_2_3:.4f}")
    print()
    print("Note: Texts 1 and 2 should be more similar (about Apple)")
    print("      than comparisons with Text 3 (about oil)")
    print()


def example_custom_configuration() -> None:
    """Demonstrate using custom embedder configuration."""
    print("=" * 80)
    print("Example 4: Custom Configuration")
    print("=" * 80)

    # Use different model for higher quality embeddings
    config = EmbeddingsConfig(
        model_name="all-mpnet-base-v2",  # Higher quality, 768 dimensions
        device="cpu",  # Force CPU usage
        normalize=True,  # L2 normalize (recommended for similarity)
        batch_size=16,
        max_length=256,
    )

    embedder = get_embedder(config)

    text = "Microsoft acquires gaming studio for $69 billion."
    result = embedder.encode(text)

    print(f"Model: {result.model_name}")
    print(f"Dimension: {result.dimension}")
    print(f"Text: {result.text}")
    print(f"Embedding (first 5): {result.embedding[:5]}")
    print()


def example_semantic_search() -> None:
    """Demonstrate simple semantic search."""
    print("=" * 80)
    print("Example 5: Semantic Search")
    print("=" * 80)

    # Document corpus
    documents = [
        "Tesla announces new Gigafactory in Texas for vehicle production.",
        "Apple unveils latest iPhone with improved camera capabilities.",
        "Federal Reserve raises interest rates to combat inflation.",
        "Amazon expands same-day delivery to 100 new cities.",
        "Google launches new AI model for natural language processing.",
        "Electric vehicle sales surge 40% in Q4 2023.",
        "Tech stocks rally on positive earnings reports.",
    ]

    # Search query
    query = "What are the latest developments in electric vehicles?"

    print(f"Query: {query}")
    print()
    print("Searching through documents...")
    print()

    # Generate embeddings
    query_emb = embed_text(query)
    doc_embs = embed_texts(documents)

    # Compute similarities
    similarities = []
    for idx, doc_emb in enumerate(doc_embs):
        sim = compute_similarity(query_emb.embedding, doc_emb.embedding)
        similarities.append((idx, sim, documents[idx]))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Display top 3 results
    print("Top 3 most relevant documents:")
    for rank, (idx, sim, doc) in enumerate(similarities[:3], 1):
        print(f"{rank}. [Similarity: {sim:.4f}] {doc}")
    print()


def example_multilingual_embedding() -> None:
    """Demonstrate multilingual embedding support."""
    print("=" * 80)
    print("Example 6: Multilingual Embeddings")
    print("=" * 80)

    # Use multilingual model
    config = EmbeddingsConfig(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        normalize=True,
    )

    embedder = get_embedder(config)

    # Same concept in different languages
    texts = [
        "The company reported strong financial results.",  # English
        "La empresa reporto resultados financieros solidos.",  # Spanish
        "Das Unternehmen meldete starke Finanzergebnisse.",  # German
    ]

    results = embedder.encode_batch(texts)

    print("Texts in different languages:")
    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result.text}")
    print()

    # Compute cross-lingual similarities
    sim_en_es = compute_similarity(results[0].embedding, results[1].embedding)
    sim_en_de = compute_similarity(results[0].embedding, results[2].embedding)
    sim_es_de = compute_similarity(results[1].embedding, results[2].embedding)

    print("Cross-lingual similarities:")
    print(f"English - Spanish: {sim_en_es:.4f}")
    print(f"English - German:  {sim_en_de:.4f}")
    print(f"Spanish - German:  {sim_es_de:.4f}")
    print()
    print("Note: Multilingual models can capture semantic similarity")
    print("      across languages despite different words.")
    print()


def main() -> None:
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  SignalForge NLP Embeddings - Example Usage".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print("\n")

    example_basic_embedding()
    example_batch_embedding()
    example_similarity_computation()
    example_custom_configuration()
    example_semantic_search()
    example_multilingual_embedding()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
