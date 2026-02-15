"""Example usage of the topic extraction module.

This script demonstrates how to use the topic extraction functionality
to identify key topics and phrases from financial text.
"""

from signalforge.nlp.topics import (
    EmbeddingTopicExtractor,
    TopicExtractionConfig,
    extract_keyphrases,
    extract_topics,
)


def main() -> None:
    """Run topic extraction examples."""
    # Sample financial text
    financial_news = """
    Apple Inc. reported strong quarterly earnings today, significantly beating
    analyst expectations. Revenue increased by 15% year-over-year, driven by
    robust iPhone sales and growing services revenue. The company's stock price
    surged 5% in after-hours trading following the announcement. CEO Tim Cook
    highlighted the strength in emerging markets and the success of their new
    product launches.
    """

    print("="*80)
    print("Topic Extraction Example")
    print("="*80)
    print("\nOriginal Text:")
    print(financial_news.strip())
    print("\n" + "="*80)

    # Example 1: Simple keyphrase extraction
    print("\n1. Simple Keyphrase Extraction (top 5):")
    print("-" * 80)
    keyphrases = extract_keyphrases(financial_news, top_n=5)
    for i, phrase in enumerate(keyphrases, 1):
        print(f"   {i}. {phrase}")

    # Example 2: Full topic extraction with scores
    print("\n2. Topic Extraction with Scores:")
    print("-" * 80)
    result = extract_topics(financial_news)
    for i, keyword in enumerate(result.keywords, 1):
        print(f"   {i}. {keyword.keyword:<40} (score: {keyword.score:.4f})")

    # Example 3: Custom configuration with MMR for diversity
    print("\n3. Topic Extraction with High Diversity (MMR):")
    print("-" * 80)
    config = TopicExtractionConfig(
        top_n=7,
        keyphrase_ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        diversity=0.7,  # High diversity
        use_mmr=True,
    )
    extractor = EmbeddingTopicExtractor(config)
    result_diverse = extractor.extract(financial_news)
    for i, keyword in enumerate(result_diverse.keywords, 1):
        print(f"   {i}. {keyword.keyword:<40} (score: {keyword.score:.4f})")

    # Example 4: Batch processing
    print("\n4. Batch Processing Multiple Articles:")
    print("-" * 80)
    articles = [
        "Tesla stock surged 8% after posting record delivery numbers.",
        "Federal Reserve maintains interest rates amid inflation concerns.",
        "Tech sector leads market rally with strong earnings reports.",
    ]

    batch_results = extractor.extract_batch(articles)
    for idx, (article, result_item) in enumerate(zip(articles, batch_results, strict=True), 1):
        print(f"\n   Article {idx}: {article[:60]}...")
        keywords_str = ", ".join([kw.keyword for kw in result_item.keywords[:3]])
        print(f"   Top keywords: {keywords_str}")

    # Example 5: Relevance-focused extraction (no diversity)
    print("\n5. Relevance-Focused Extraction (No Diversity):")
    print("-" * 80)
    config_relevant = TopicExtractionConfig(
        top_n=5,
        use_mmr=False,  # Disable MMR for pure relevance ranking
    )
    extractor_relevant = EmbeddingTopicExtractor(config_relevant)
    result_relevant = extractor_relevant.extract(financial_news)
    for i, keyword in enumerate(result_relevant.keywords, 1):
        print(f"   {i}. {keyword.keyword:<40} (score: {keyword.score:.4f})")

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
