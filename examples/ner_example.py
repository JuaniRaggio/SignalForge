"""Example usage of the NER module for financial text.

This script demonstrates how to use the Named Entity Recognition module
to extract entities from financial news and reports.

Note: This example requires spaCy and the en_core_web_sm model to be installed:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

from signalforge.nlp.ner import (
    FinancialEntityExtractor,
    NERConfig,
    extract_entities,
    extract_tickers,
    get_entity_extractor,
)


def basic_entity_extraction() -> None:
    """Demonstrate basic entity extraction."""
    print("=" * 80)
    print("Basic Entity Extraction")
    print("=" * 80)

    text = """
    Apple Inc. (AAPL) reported quarterly revenue of $1.5B, up 15% from last year.
    CEO Tim Cook announced the results from the company's Cupertino headquarters.
    The stock rose $5.25 to reach $150.75 per share on the news.
    """

    result = extract_entities(text)

    print(f"\nText: {text.strip()}\n")
    print(f"Found {len(result.entities)} entities:")
    for entity in result.entities:
        print(f"  - {entity.text:20} | {entity.label:10} | Confidence: {entity.confidence:.2f}")

    print(f"\nEntity counts: {result.entity_counts}")


def ticker_extraction() -> None:
    """Demonstrate ticker symbol extraction."""
    print("\n" + "=" * 80)
    print("Ticker Symbol Extraction")
    print("=" * 80)

    text = """
    Trading activity was high today with $AAPL, MSFT, and GOOGL all moving up.
    Meanwhile, TSLA declined 3% and NVDA remained flat. The tech sector (AAPL, MSFT)
    led the market gains.
    """

    tickers = extract_tickers(text)

    print(f"\nText: {text.strip()}\n")
    print(f"Extracted tickers: {', '.join(tickers)}")
    print(f"Total unique tickers: {len(tickers)}")


def batch_extraction() -> None:
    """Demonstrate batch entity extraction."""
    print("\n" + "=" * 80)
    print("Batch Entity Extraction")
    print("=" * 80)

    texts = [
        "Microsoft reported revenue of $50B, up 12% year-over-year.",
        "Tesla's stock price jumped 20% after the earnings announcement.",
        "JPMorgan Chase increased its dividend by $0.25 per share.",
    ]

    config = NERConfig(
        include_custom_patterns=True,
        confidence_threshold=0.5,
        batch_size=32,
    )
    extractor = get_entity_extractor(config)
    results = extractor.extract_batch(texts)

    for i, result in enumerate(results, 1):
        print(f"\nText {i}: {result.text[:60]}...")
        print(f"  Entities found: {len(result.entities)}")
        for entity in result.entities:
            print(f"    - {entity.text:15} | {entity.label:10}")


def custom_configuration() -> None:
    """Demonstrate custom NER configuration."""
    print("\n" + "=" * 80)
    print("Custom Configuration")
    print("=" * 80)

    # Create custom config with higher confidence threshold
    config = NERConfig(
        model_name="en_core_web_sm",
        include_custom_patterns=True,
        confidence_threshold=0.8,  # Higher threshold for more precision
        batch_size=16,
    )

    extractor = FinancialEntityExtractor(config=config)

    text = """
    Goldman Sachs analysts raised their price target for Amazon.com Inc. (AMZN)
    to $175, citing strong cloud revenue growth of 25%. The bank expects AWS
    revenue to reach $100B by 2025.
    """

    result = extractor.extract(text)

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print(f"  Custom patterns: {config.include_custom_patterns}")

    print(f"\nText: {text.strip()}\n")
    print(f"Extracted {len(result.entities)} high-confidence entities:")
    for entity in result.entities:
        print(f"  - {entity.text:20} | {entity.label:10} | {entity.confidence:.2f}")


def entity_type_filtering() -> None:
    """Demonstrate filtering entities by type."""
    print("\n" + "=" * 80)
    print("Entity Type Filtering")
    print("=" * 80)

    text = """
    Apple Inc., Microsoft Corp., and Alphabet Inc. are the three largest tech
    companies by market cap. Apple is based in Cupertino, Microsoft in Redmond,
    and Alphabet in Mountain View. All three reported earnings on January 31, 2024.
    """

    result = extract_entities(text)

    print(f"\nText: {text.strip()}\n")

    # Filter by entity type
    orgs = [e for e in result.entities if e.label == "ORG"]
    locations = [e for e in result.entities if e.label == "GPE"]
    dates = [e for e in result.entities if e.label == "DATE"]

    print(f"Organizations ({len(orgs)}):")
    for entity in orgs:
        print(f"  - {entity.text}")

    print(f"\nLocations ({len(locations)}):")
    for entity in locations:
        print(f"  - {entity.text}")

    print(f"\nDates ({len(dates)}):")
    for entity in dates:
        print(f"  - {entity.text}")


def main() -> None:
    """Run all examples."""
    print("\nNamed Entity Recognition Examples for Financial Text\n")

    try:
        basic_entity_extraction()
        ticker_extraction()
        batch_extraction()
        custom_configuration()
        entity_type_filtering()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed spaCy and the required model:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    main()
