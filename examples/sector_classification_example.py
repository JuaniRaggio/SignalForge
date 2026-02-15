"""Example usage of the GICS sector classifier.

This script demonstrates how to use the sector classification module to
classify financial documents into GICS sectors.
"""

from signalforge.nlp.sector_classifier import (
    SectorClassifierConfig,
    classify_sector,
    get_all_sectors,
    get_sector_classifier,
)


def example_basic_classification() -> None:
    """Example of basic sector classification using convenience function."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Sector Classification")
    print("=" * 80)

    texts = [
        "Apple Inc. reported strong iPhone sales and growing cloud services revenue.",
        "Goldman Sachs investment banking division saw increased M&A activity.",
        "ExxonMobil announced plans to expand oil drilling operations in the Gulf.",
        "Pfizer's new cancer drug received FDA approval for clinical trials.",
        "Amazon Web Services launches new AI-powered cloud infrastructure.",
    ]

    for text in texts:
        prediction = classify_sector(text)
        print(f"\nText: {text[:70]}...")
        print(f"Sector: {prediction.sector}")
        print(f"Confidence: {prediction.confidence:.3f}")

        # Show top 3 sectors
        top_3 = prediction.get_top_k_sectors(3)
        print("Top 3 sectors:")
        for sector, score in top_3:
            print(f"  - {sector}: {score:.3f}")


def example_batch_classification() -> None:
    """Example of batch classification with custom configuration."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Classification with Custom Config")
    print("=" * 80)

    # Configure classifier with custom parameters
    config = SectorClassifierConfig(
        similarity_threshold=0.4,
        top_k=5,
    )

    classifier = get_sector_classifier(config)

    texts = [
        "JPMorgan Chase posted record quarterly profits from lending operations.",
        "Tesla delivered more electric vehicles than expected last quarter.",
        "Microsoft Azure cloud platform continues to gain market share.",
        "ConocoPhillips increases natural gas production capacity.",
        "Johnson & Johnson settles pharmaceutical liability claims.",
    ]

    predictions = classifier.classify_batch(texts)

    for text, prediction in zip(texts, predictions, strict=True):
        print(f"\nText: {text}")
        print(f"Sector: {prediction.sector} (confidence: {prediction.confidence:.3f})")

        # Check if confidence meets threshold
        threshold_met = prediction.metadata.get("threshold_met", False)
        if not threshold_met:
            print("  Warning: Confidence below threshold!")


def example_sector_scores() -> None:
    """Example of analyzing all sector scores."""
    print("\n" + "=" * 80)
    print("Example 3: Analyzing All Sector Scores")
    print("=" * 80)

    text = "Semiconductor manufacturer announces new chip design for AI workloads."

    prediction = classify_sector(text)

    print(f"\nText: {text}")
    print(f"\nPrimary Sector: {prediction.sector} ({prediction.confidence:.3f})")
    print("\nAll Sector Scores:")

    # Sort sectors by score
    sorted_scores = sorted(
        prediction.all_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for sector, score in sorted_scores:
        bar = "=" * int(score * 50)  # Visual bar chart
        print(f"  {sector:30s} {score:.3f} {bar}")


def example_list_sectors() -> None:
    """Example of listing all available GICS sectors."""
    print("\n" + "=" * 80)
    print("Example 4: List All GICS Sectors")
    print("=" * 80)

    sectors = get_all_sectors()

    print(f"\nTotal GICS Sectors: {len(sectors)}\n")
    for i, sector in enumerate(sectors, 1):
        print(f"  {i:2d}. {sector}")


def main() -> None:
    """Run all examples."""
    print("\n" + "#" * 80)
    print("# SignalForge Sector Classification Examples")
    print("#" * 80)

    try:
        example_list_sectors()
        example_basic_classification()
        example_batch_classification()
        example_sector_scores()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: Make sure sentence-transformers is installed:")
        print("  pip install sentence-transformers torch")


if __name__ == "__main__":
    main()
