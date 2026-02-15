"""Demonstration of the NLP preprocessing pipeline.

This script shows how to use the NLP preprocessing components for
financial text analysis.
"""

from signalforge.nlp.preprocessing import (
    DocumentPreprocessor,
    PreprocessingConfig,
    TextPreprocessor,
)


def demo_text_preprocessor() -> None:
    """Demonstrate TextPreprocessor capabilities."""
    print("=" * 80)
    print("TextPreprocessor Demo")
    print("=" * 80)

    preprocessor = TextPreprocessor()

    # Example 1: Clean text
    raw_text = """  Apple Inc. (AAPL) reported revenue of $1.5M in Q4 2024.
    Visit https://investor.apple.com for more details.  """
    print("\n1. Text Cleaning:")
    print(f"   Original: {raw_text!r}")
    cleaned = preprocessor.clean_text(raw_text)
    print(f"   Cleaned:  {cleaned!r}")

    # Example 2: Normalize financial terms
    financial_text = "Revenue increased to 2.5B dollars on 12/31/2024"
    print("\n2. Financial Term Normalization:")
    print(f"   Original:   {financial_text}")
    normalized = preprocessor.normalize_financial_terms(financial_text)
    print(f"   Normalized: {normalized}")

    # Example 3: Sentence extraction
    multi_sentence = "Revenue increased. EPS was $2.50. Growth continues."
    print("\n3. Sentence Extraction:")
    print(f"   Original: {multi_sentence}")
    sentences = preprocessor.extract_sentences(multi_sentence)
    for i, sent in enumerate(sentences, 1):
        print(f"   Sentence {i}: {sent}")

    # Example 4: Tokenization
    tokenize_text = "Apple's revenue was $100M (up 15%)"
    print("\n4. Tokenization:")
    print(f"   Original: {tokenize_text}")
    tokens = preprocessor.tokenize(tokenize_text)
    print(f"   Tokens:   {tokens}")


def demo_document_preprocessor() -> None:
    """Demonstrate DocumentPreprocessor capabilities."""
    print("\n" + "=" * 80)
    print("DocumentPreprocessor Demo")
    print("=" * 80)

    preprocessor = DocumentPreprocessor()

    # Example document
    document = """
    Apple Inc. (AAPL) Reports Record Q4 2024 Results

    CUPERTINO, California - Apple Inc. today announced financial results for
    its fiscal 2024 fourth quarter ended September 28, 2024. Revenue was $1.5B,
    up 15% year-over-year.

    "We are pleased with these outstanding results," said CEO Tim Cook.

    Forward-looking statements: This press release contains forward-looking
    statements that involve risks and uncertainties.

    For more information, contact:
    Investor Relations
    investor@apple.com
    """

    # Configuration 1: Standard processing with boilerplate removal
    print("\n1. Standard Processing (with boilerplate removal):")
    config_standard = PreprocessingConfig(
        lowercase=True,
        remove_boilerplate=True,
        normalize_financial_terms=True,
    )
    doc = preprocessor.process_document(document, config_standard)
    print(f"   Original length: {doc.metadata['original_length']} chars")
    print(f"   Cleaned length:  {doc.metadata['cleaned_length']} chars")
    print(f"   Sentences:       {doc.metadata['num_sentences']}")
    print(f"   Tokens:          {doc.metadata['num_tokens']}")
    print(f"\n   First sentence:  {doc.sentences[0] if doc.sentences else 'N/A'}")

    # Configuration 2: Preserve case, no boilerplate removal
    print("\n2. Preserve Case (no boilerplate removal):")
    config_preserve = PreprocessingConfig(
        lowercase=False,
        remove_boilerplate=False,
        normalize_financial_terms=True,
    )
    doc2 = preprocessor.process_document(document, config_preserve)
    print(f"   Sentences:       {doc2.metadata['num_sentences']}")
    print(f"   First sentence:  {doc2.sentences[0][:80] if doc2.sentences else 'N/A'}...")

    # Configuration 3: Token filtering
    print("\n3. Token Filtering:")
    config_filtered = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=True,
        min_token_length=3,
    )
    doc3 = preprocessor.process_document(document, config_filtered)
    print(f"   Tokens (filtered): {doc3.metadata['num_tokens']}")
    print(f"   Sample tokens:     {doc3.tokens[:10]}")


def demo_batch_processing() -> None:
    """Demonstrate batch document processing."""
    print("\n" + "=" * 80)
    print("Batch Processing Demo")
    print("=" * 80)

    preprocessor = DocumentPreprocessor()
    config = PreprocessingConfig()

    documents = [
        "Apple Inc. reported revenue of $1.5M in Q4.",
        "Tesla's deliveries increased 50% year-over-year to 500K units.",
        "Microsoft announced a dividend of $0.75 per share.",
        "",  # Empty document
        "Amazon Web Services revenue grew to 100B dollars.",
    ]

    print(f"\nProcessing {len(documents)} documents...")
    results = preprocessor.process_batch(documents, config)

    print(f"\nResults:")
    for i, doc in enumerate(results, 1):
        status = "Success" if doc.cleaned_text else "Empty/Failed"
        print(f"   Doc {i}: {status:12s} - {doc.metadata['num_tokens']:3d} tokens")


if __name__ == "__main__":
    demo_text_preprocessor()
    demo_document_preprocessor()
    demo_batch_processing()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
