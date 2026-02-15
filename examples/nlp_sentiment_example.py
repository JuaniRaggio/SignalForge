"""Example usage of the Sentiment Analysis module.

This script demonstrates how to use the FinBERT-based sentiment analyzer
for financial text analysis.

Requirements:
    - transformers>=4.36.0
    - torch>=2.1.0

Usage:
    python examples/nlp_sentiment_example.py
"""

from signalforge.nlp.sentiment import (
    FinBERTSentimentAnalyzer,
    SentimentConfig,
    analyze_financial_text,
    get_sentiment_analyzer,
)


def basic_usage_example() -> None:
    """Demonstrate basic sentiment analysis."""
    print("=== Basic Usage Example ===\n")

    # Analyze a single text using the convenience function
    text = "Apple Inc. reported strong quarterly earnings, beating analyst expectations."
    result = analyze_financial_text(text)

    print(f"Text: {result.text}")
    print(f"Sentiment: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Scores: {result.scores}\n")


def batch_processing_example() -> None:
    """Demonstrate batch sentiment analysis."""
    print("=== Batch Processing Example ===\n")

    # Create an analyzer instance
    analyzer = get_sentiment_analyzer()

    # Analyze multiple texts
    texts = [
        "Revenue exceeded expectations, driving strong market performance.",
        "The company announced significant layoffs amid declining sales.",
        "Quarterly results were in line with market forecasts.",
        "Stock price surged following the merger announcement.",
        "Investors concerned about mounting debt and cash flow issues.",
    ]

    results = analyzer.analyze_batch(texts)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.text[:60]}...")
        print(f"   Sentiment: {result.label} ({result.confidence:.2%})")
        print()


def custom_configuration_example() -> None:
    """Demonstrate custom configuration."""
    print("=== Custom Configuration Example ===\n")

    # Create a custom configuration
    config = SentimentConfig(
        device="cpu",  # Force CPU usage
        batch_size=4,  # Process 4 texts at a time
        max_length=256,  # Truncate to 256 tokens
        preprocess_text=True,  # Enable text preprocessing
        temperature=0.8,  # Apply temperature scaling
    )

    # Create analyzer with custom config
    analyzer = FinBERTSentimentAnalyzer(config)

    # Analyze text
    text = "  Profits soared this quarter!  https://example.com  "
    result = analyzer.analyze(text)

    print(f"Original text: '{text}'")
    print(f"Sentiment: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    print("Note: URL was removed during preprocessing\n")


def detailed_scores_example() -> None:
    """Demonstrate accessing detailed sentiment scores."""
    print("=== Detailed Scores Example ===\n")

    analyzer = get_sentiment_analyzer()

    text = "The Federal Reserve announced an interest rate hike."
    result = analyzer.analyze(text)

    print(f"Text: {result.text}\n")
    print("Detailed Scores:")
    for label, score in sorted(result.scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label.capitalize()}: {score:.2%}")
    print(f"\nPredicted: {result.label} (confidence: {result.confidence:.2%})\n")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SignalForge - Sentiment Analysis Examples")
    print("=" * 70 + "\n")

    try:
        basic_usage_example()
        batch_processing_example()
        custom_configuration_example()
        detailed_scores_example()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("\nPlease install required packages:")
        print("  pip install transformers torch")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
