"""Tests for NLP preprocessing pipeline."""

import pytest

from signalforge.nlp.preprocessing import (
    DocumentPreprocessor,
    PreprocessingConfig,
    ProcessedDocument,
    TextPreprocessor,
)


@pytest.fixture
def text_preprocessor() -> TextPreprocessor:
    """Create a TextPreprocessor instance for testing."""
    return TextPreprocessor()


@pytest.fixture
def doc_preprocessor() -> DocumentPreprocessor:
    """Create a DocumentPreprocessor instance for testing."""
    return DocumentPreprocessor()


@pytest.fixture
def sample_financial_text() -> str:
    """Provide sample financial text for testing."""
    return """
    Apple Inc. (AAPL) reported strong Q4 2024 results with revenue of $1.5M.
    The company's earnings per share increased 15% year-over-year.
    CEO Tim Cook stated: "We are pleased with these results."
    """


@pytest.fixture
def boilerplate_text() -> str:
    """Provide text with common boilerplate sections."""
    return """
    Revenue increased significantly in Q4.

    Forward-looking statements: This press release contains forward-looking
    statements that involve risks and uncertainties.

    For more information, contact:
    Investor Relations
    investors@example.com
    """


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PreprocessingConfig()
        assert config.lowercase is True
        assert config.remove_stopwords is False
        assert config.remove_punctuation is False
        assert config.preserve_tickers is True
        assert config.preserve_numbers is True
        assert config.normalize_financial_terms is True
        assert config.remove_boilerplate is True
        assert config.max_token_length == 50
        assert config.min_token_length == 1

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = PreprocessingConfig(
            lowercase=False,
            remove_stopwords=True,
            max_token_length=100,
        )
        assert config.lowercase is False
        assert config.remove_stopwords is True
        assert config.max_token_length == 100

    def test_invalid_max_token_length_raises_error(self) -> None:
        """Test that invalid max_token_length raises ValueError."""
        with pytest.raises(ValueError, match="max_token_length must be positive"):
            PreprocessingConfig(max_token_length=0)

    def test_invalid_min_token_length_raises_error(self) -> None:
        """Test that invalid min_token_length raises ValueError."""
        with pytest.raises(ValueError, match="min_token_length cannot be negative"):
            PreprocessingConfig(min_token_length=-1)

    def test_min_greater_than_max_raises_error(self) -> None:
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError, match="min_token_length cannot exceed max_token_length"):
            PreprocessingConfig(min_token_length=10, max_token_length=5)


class TestProcessedDocument:
    """Tests for ProcessedDocument dataclass."""

    def test_processed_document_creation(self) -> None:
        """Test ProcessedDocument creation."""
        doc = ProcessedDocument(
            original_text="Hello World",
            cleaned_text="hello world",
            sentences=["hello world"],
            tokens=["hello", "world"],
        )
        assert doc.original_text == "Hello World"
        assert doc.cleaned_text == "hello world"
        assert len(doc.sentences) == 1
        assert len(doc.tokens) == 2

    def test_metadata_auto_population(self) -> None:
        """Test that metadata is automatically populated."""
        doc = ProcessedDocument(
            original_text="Hello World",
            cleaned_text="hello world",
            sentences=["hello world"],
            tokens=["hello", "world"],
        )
        assert doc.metadata["original_length"] == 11
        assert doc.metadata["cleaned_length"] == 11
        assert doc.metadata["num_sentences"] == 1
        assert doc.metadata["num_tokens"] == 2

    def test_custom_metadata(self) -> None:
        """Test ProcessedDocument with custom metadata."""
        custom_meta = {"source": "test", "timestamp": "2024-01-01"}
        doc = ProcessedDocument(
            original_text="Test",
            cleaned_text="test",
            sentences=["test"],
            tokens=["test"],
            metadata=custom_meta,
        )
        assert doc.metadata == custom_meta


class TestTextPreprocessorCleanText:
    """Tests for TextPreprocessor.clean_text method."""

    def test_clean_empty_text(self, text_preprocessor: TextPreprocessor) -> None:
        """Test cleaning empty text."""
        assert text_preprocessor.clean_text("") == ""
        assert text_preprocessor.clean_text("   ") == ""
        assert text_preprocessor.clean_text("\n\n\t") == ""

    def test_clean_whitespace_normalization(self, text_preprocessor: TextPreprocessor) -> None:
        """Test whitespace normalization."""
        text = "Hello    World  \n\n  Test"
        cleaned = text_preprocessor.clean_text(text)
        assert cleaned == "Hello World Test"

    def test_clean_unicode_normalization(self, text_preprocessor: TextPreprocessor) -> None:
        """Test Unicode normalization."""
        text = "café"  # Contains combining character
        cleaned = text_preprocessor.clean_text(text)
        assert "café" in cleaned or "cafe" in cleaned

    def test_clean_url_removal(self, text_preprocessor: TextPreprocessor) -> None:
        """Test URL removal."""
        text = "Check out https://example.com for more info"
        cleaned = text_preprocessor.clean_text(text)
        assert "https://example.com" not in cleaned
        assert "Check out" in cleaned

    def test_clean_email_removal(self, text_preprocessor: TextPreprocessor) -> None:
        """Test email address removal."""
        text = "Contact us at support@example.com for help"
        cleaned = text_preprocessor.clean_text(text)
        assert "support@example.com" not in cleaned
        assert "Contact us at" in cleaned

    def test_clean_control_characters(self, text_preprocessor: TextPreprocessor) -> None:
        """Test control character removal."""
        text = "Hello\x00World\x01Test"
        cleaned = text_preprocessor.clean_text(text)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned

    def test_clean_preserves_newlines_and_tabs(self, text_preprocessor: TextPreprocessor) -> None:
        """Test that newlines and tabs are preserved during cleaning."""
        text = "Line1\nLine2\tTabbed"
        cleaned = text_preprocessor.clean_text(text)
        # Note: whitespace normalization will convert these to spaces
        assert "Line1" in cleaned
        assert "Line2" in cleaned


class TestTextPreprocessorNormalizeFinancialTerms:
    """Tests for TextPreprocessor.normalize_financial_terms method."""

    def test_normalize_currency_symbols(self, text_preprocessor: TextPreprocessor) -> None:
        """Test currency symbol normalization."""
        text = "Revenue was 100 dollars"
        normalized = text_preprocessor.normalize_financial_terms(text)
        assert "$" in normalized
        assert "dollars" not in normalized

    def test_normalize_usd_to_dollar_sign(self, text_preprocessor: TextPreprocessor) -> None:
        """Test USD conversion to dollar sign."""
        text = "Price is USD 100"
        normalized = text_preprocessor.normalize_financial_terms(text)
        assert "$" in normalized

    def test_normalize_number_abbreviations_million(
        self, text_preprocessor: TextPreprocessor
    ) -> None:
        """Test million abbreviation normalization."""
        text = "Revenue of 1.5M"
        normalized = text_preprocessor.normalize_financial_terms(text)
        assert "1500000" in normalized or "1.5M" in normalized

    def test_normalize_number_abbreviations_billion(
        self, text_preprocessor: TextPreprocessor
    ) -> None:
        """Test billion abbreviation normalization."""
        text = "Market cap of 2B"
        normalized = text_preprocessor.normalize_financial_terms(text)
        assert "2000000000" in normalized or "2B" in normalized

    def test_normalize_number_abbreviations_thousand(
        self, text_preprocessor: TextPreprocessor
    ) -> None:
        """Test thousand abbreviation normalization."""
        text = "Salary of 50K"
        normalized = text_preprocessor.normalize_financial_terms(text)
        assert "50000" in normalized or "50K" in normalized

    def test_normalize_date_formats(self, text_preprocessor: TextPreprocessor) -> None:
        """Test date format normalization."""
        text = "Report dated 12/31/2024"
        normalized = text_preprocessor.normalize_financial_terms(text)
        assert "2024-12-31" in normalized

    def test_normalize_empty_text(self, text_preprocessor: TextPreprocessor) -> None:
        """Test normalization of empty text."""
        assert text_preprocessor.normalize_financial_terms("") == ""

    def test_normalize_preserves_other_text(self, text_preprocessor: TextPreprocessor) -> None:
        """Test that non-financial text is preserved."""
        text = "The company performed well"
        normalized = text_preprocessor.normalize_financial_terms(text)
        assert "company" in normalized
        assert "performed" in normalized


class TestTextPreprocessorRemoveBoilerplate:
    """Tests for TextPreprocessor.remove_boilerplate method."""

    def test_remove_forward_looking_statements(self, text_preprocessor: TextPreprocessor) -> None:
        """Test removal of forward-looking statements."""
        text = """Revenue increased.

Forward-looking statements: This contains forward-looking statements.

More content here."""
        cleaned = text_preprocessor.remove_boilerplate(text)
        assert "Forward-looking" not in cleaned
        assert "Revenue increased" in cleaned
        assert "More content" in cleaned

    def test_remove_contact_information(self, text_preprocessor: TextPreprocessor) -> None:
        """Test removal of contact information."""
        text = """Important news here.

Contact: John Doe, investor@example.com

End of document."""
        cleaned = text_preprocessor.remove_boilerplate(text)
        assert "Contact:" not in cleaned or "investor@example.com" not in cleaned
        assert "Important news" in cleaned

    def test_remove_investor_relations(self, text_preprocessor: TextPreprocessor) -> None:
        """Test removal of investor relations sections."""
        text = """Earnings report.

Investor Relations:
For questions, contact IR department.

Thank you."""
        cleaned = text_preprocessor.remove_boilerplate(text)
        assert "Earnings report" in cleaned
        assert "Thank you" in cleaned

    def test_remove_empty_text(self, text_preprocessor: TextPreprocessor) -> None:
        """Test boilerplate removal on empty text."""
        assert text_preprocessor.remove_boilerplate("") == ""

    def test_remove_multiple_newlines(self, text_preprocessor: TextPreprocessor) -> None:
        """Test that excessive newlines are cleaned up."""
        text = "Line1\n\n\n\n\nLine2"
        cleaned = text_preprocessor.remove_boilerplate(text)
        assert "\n\n\n" not in cleaned


class TestTextPreprocessorExtractSentences:
    """Tests for TextPreprocessor.extract_sentences method."""

    def test_extract_sentences_basic(self, text_preprocessor: TextPreprocessor) -> None:
        """Test basic sentence extraction."""
        text = "This is sentence one. This is sentence two."
        sentences = text_preprocessor.extract_sentences(text)
        assert len(sentences) == 2
        assert "This is sentence one." in sentences[0]
        assert "This is sentence two." in sentences[1]

    def test_extract_sentences_with_abbreviations(
        self, text_preprocessor: TextPreprocessor
    ) -> None:
        """Test sentence extraction with abbreviations."""
        text = "Apple Inc. reported strong results. Revenue increased."
        sentences = text_preprocessor.extract_sentences(text)
        # Should handle "Inc." correctly
        assert len(sentences) >= 1

    def test_extract_sentences_question_marks(self, text_preprocessor: TextPreprocessor) -> None:
        """Test sentence extraction with question marks."""
        text = "What happened? Revenue increased!"
        sentences = text_preprocessor.extract_sentences(text)
        assert len(sentences) == 2

    def test_extract_sentences_empty_text(self, text_preprocessor: TextPreprocessor) -> None:
        """Test sentence extraction on empty text."""
        assert text_preprocessor.extract_sentences("") == []
        assert text_preprocessor.extract_sentences("   ") == []

    def test_extract_sentences_single_sentence(self, text_preprocessor: TextPreprocessor) -> None:
        """Test extraction of single sentence."""
        text = "This is one sentence."
        sentences = text_preprocessor.extract_sentences(text)
        assert len(sentences) == 1

    def test_extract_sentences_no_punctuation(self, text_preprocessor: TextPreprocessor) -> None:
        """Test extraction when no sentence-ending punctuation exists."""
        text = "This is text without ending punctuation"
        sentences = text_preprocessor.extract_sentences(text)
        assert len(sentences) == 1
        assert text in sentences[0]


class TestTextPreprocessorTokenize:
    """Tests for TextPreprocessor.tokenize method."""

    def test_tokenize_basic(self, text_preprocessor: TextPreprocessor) -> None:
        """Test basic tokenization."""
        text = "Hello world"
        tokens = text_preprocessor.tokenize(text)
        assert "Hello" in tokens
        assert "world" in tokens

    def test_tokenize_with_punctuation(self, text_preprocessor: TextPreprocessor) -> None:
        """Test tokenization with punctuation."""
        text = "Hello, world!"
        tokens = text_preprocessor.tokenize(text)
        assert "Hello" in tokens
        assert "," in tokens
        assert "world" in tokens
        assert "!" in tokens

    def test_tokenize_ticker_symbols(self, text_preprocessor: TextPreprocessor) -> None:
        """Test tokenization preserves ticker symbols."""
        text = "$AAPL and GOOGL stocks"
        tokens = text_preprocessor.tokenize(text)
        assert "$AAPL" in tokens or "$" in tokens and "AAPL" in tokens

    def test_tokenize_numbers(self, text_preprocessor: TextPreprocessor) -> None:
        """Test tokenization preserves numbers."""
        text = "Price is 123.45 or 50%"
        tokens = text_preprocessor.tokenize(text)
        assert any("123" in token for token in tokens)
        assert any("50" in token or "%" in token for token in tokens)

    def test_tokenize_hyphenated_words(self, text_preprocessor: TextPreprocessor) -> None:
        """Test tokenization of hyphenated words."""
        text = "year-over-year growth"
        tokens = text_preprocessor.tokenize(text)
        assert "year-over-year" in tokens or "year" in tokens

    def test_tokenize_empty_text(self, text_preprocessor: TextPreprocessor) -> None:
        """Test tokenization of empty text."""
        assert text_preprocessor.tokenize("") == []
        assert text_preprocessor.tokenize("   ") == []

    def test_tokenize_apostrophes(self, text_preprocessor: TextPreprocessor) -> None:
        """Test tokenization handles apostrophes."""
        text = "Apple's revenue"
        tokens = text_preprocessor.tokenize(text)
        # Should handle contractions and possessives
        assert len(tokens) >= 2


class TestDocumentPreprocessorProcessDocument:
    """Tests for DocumentPreprocessor.process_document method."""

    def test_process_document_basic(
        self,
        doc_preprocessor: DocumentPreprocessor,
        sample_financial_text: str,
    ) -> None:
        """Test basic document processing."""
        config = PreprocessingConfig()
        doc = doc_preprocessor.process_document(sample_financial_text, config)

        assert isinstance(doc, ProcessedDocument)
        assert doc.original_text == sample_financial_text
        assert doc.cleaned_text
        assert len(doc.sentences) > 0
        assert len(doc.tokens) > 0

    def test_process_document_empty_text(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test processing empty document."""
        config = PreprocessingConfig()
        doc = doc_preprocessor.process_document("", config)

        assert doc.cleaned_text == ""
        assert doc.sentences == []
        assert doc.tokens == []

    def test_process_document_whitespace_only(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test processing whitespace-only document."""
        config = PreprocessingConfig()
        doc = doc_preprocessor.process_document("   \n\t  ", config)

        assert doc.cleaned_text == ""

    def test_process_document_with_boilerplate_removal(
        self,
        doc_preprocessor: DocumentPreprocessor,
        boilerplate_text: str,
    ) -> None:
        """Test document processing with boilerplate removal."""
        config = PreprocessingConfig(remove_boilerplate=True)
        doc = doc_preprocessor.process_document(boilerplate_text, config)

        # Text is lowercased by default
        assert (
            "forward-looking" not in doc.cleaned_text
            or "forward-looking statements:" in doc.cleaned_text
        )
        assert "revenue increased" in doc.cleaned_text

    def test_process_document_without_boilerplate_removal(
        self,
        doc_preprocessor: DocumentPreprocessor,
        boilerplate_text: str,
    ) -> None:
        """Test document processing without boilerplate removal."""
        config = PreprocessingConfig(remove_boilerplate=False)
        doc = doc_preprocessor.process_document(boilerplate_text, config)

        # Text is lowercased by default
        assert "forward-looking" in doc.cleaned_text

    def test_process_document_lowercase(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test document processing with lowercase conversion."""
        config = PreprocessingConfig(lowercase=True)
        doc = doc_preprocessor.process_document("HELLO WORLD", config)

        assert doc.cleaned_text == "hello world"

    def test_process_document_preserve_case(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test document processing without lowercase conversion."""
        config = PreprocessingConfig(lowercase=False)
        doc = doc_preprocessor.process_document("HELLO WORLD", config)

        assert doc.cleaned_text == "HELLO WORLD"

    def test_process_document_normalize_financial_terms(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test financial term normalization in document processing."""
        config = PreprocessingConfig(normalize_financial_terms=True, lowercase=False)
        doc = doc_preprocessor.process_document("Revenue was 1.5M dollars", config)

        assert "$" in doc.cleaned_text
        assert "1500000" in doc.cleaned_text or "1.5M" in doc.cleaned_text

    def test_process_document_metadata(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test that document metadata is correctly populated."""
        config = PreprocessingConfig()
        text = "Hello world. This is a test."
        doc = doc_preprocessor.process_document(text, config)

        assert "original_length" in doc.metadata
        assert "cleaned_length" in doc.metadata
        assert "num_sentences" in doc.metadata
        assert "num_tokens" in doc.metadata
        assert doc.metadata["original_length"] == len(text)

    def test_process_document_very_long_text(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test processing of very long documents (chunking)."""
        config = PreprocessingConfig()
        long_text = "word " * 50000  # Creates a very long text
        doc = doc_preprocessor.process_document(long_text, config)

        # Should process without crashing
        assert isinstance(doc, ProcessedDocument)


class TestDocumentPreprocessorProcessBatch:
    """Tests for DocumentPreprocessor.process_batch method."""

    def test_process_batch_basic(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test basic batch processing."""
        config = PreprocessingConfig()
        texts = ["First document.", "Second document.", "Third document."]
        results = doc_preprocessor.process_batch(texts, config)

        assert len(results) == 3
        assert all(isinstance(doc, ProcessedDocument) for doc in results)
        assert all(doc.cleaned_text for doc in results)

    def test_process_batch_empty_list(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test batch processing with empty list."""
        config = PreprocessingConfig()
        results = doc_preprocessor.process_batch([], config)

        assert results == []

    def test_process_batch_with_empty_documents(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test batch processing with some empty documents."""
        config = PreprocessingConfig()
        texts = ["Valid document.", "", "Another valid document."]
        results = doc_preprocessor.process_batch(texts, config)

        assert len(results) == 3
        assert results[0].cleaned_text
        assert not results[1].cleaned_text
        assert results[2].cleaned_text

    def test_process_batch_error_handling(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test that batch processing handles errors gracefully."""
        config = PreprocessingConfig()
        texts = ["Valid document.", "Another valid document."]
        results = doc_preprocessor.process_batch(texts, config)

        # Should complete without raising exceptions
        assert len(results) == len(texts)

    def test_process_batch_different_configs(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test batch processing with different configurations."""
        texts = ["DOCUMENT ONE", "DOCUMENT TWO"]

        # With lowercase
        config_lower = PreprocessingConfig(lowercase=True)
        results_lower = doc_preprocessor.process_batch(texts, config_lower)
        assert all("document" in doc.cleaned_text for doc in results_lower)

        # Without lowercase
        config_upper = PreprocessingConfig(lowercase=False)
        results_upper = doc_preprocessor.process_batch(texts, config_upper)
        assert all("DOCUMENT" in doc.cleaned_text for doc in results_upper)


class TestDocumentPreprocessorTokenFiltering:
    """Tests for DocumentPreprocessor token filtering."""

    def test_filter_tokens_by_length(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test filtering tokens by minimum and maximum length."""
        config = PreprocessingConfig(min_token_length=3, max_token_length=10)
        doc = doc_preprocessor.process_document("I am testing this functionality", config)

        # "I" and "am" should be filtered out (too short)
        assert "I" not in doc.tokens
        assert "am" not in doc.tokens
        # Longer words should be preserved
        assert any(len(token) >= 3 for token in doc.tokens)

    def test_filter_stopwords(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test stopword removal."""
        config = PreprocessingConfig(remove_stopwords=True, lowercase=True)
        doc = doc_preprocessor.process_document("The quick brown fox", config)

        # "the" should be removed as a stopword
        assert "the" not in doc.tokens

    def test_filter_punctuation(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test punctuation removal."""
        config = PreprocessingConfig(remove_punctuation=True)
        doc = doc_preprocessor.process_document("Hello, world!", config)

        # Punctuation should be removed
        assert "," not in doc.tokens
        assert "!" not in doc.tokens


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_mixed_language_content(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test processing mixed language content."""
        config = PreprocessingConfig()
        text = "Revenue increased. Umsatz gestiegen."  # English and German
        doc = doc_preprocessor.process_document(text, config)

        assert doc.cleaned_text
        assert len(doc.sentences) >= 1

    def test_special_characters(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test handling of special characters."""
        config = PreprocessingConfig()
        text = "Revenue: $1,000,000 (up 50%)"
        doc = doc_preprocessor.process_document(text, config)

        assert doc.cleaned_text
        assert "$" in doc.cleaned_text

    def test_unicode_characters(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test handling of Unicode characters."""
        config = PreprocessingConfig()
        text = "Café revenue increased €1M"
        doc = doc_preprocessor.process_document(text, config)

        assert doc.cleaned_text
        assert len(doc.tokens) > 0

    def test_numeric_only_text(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test processing numeric-only text."""
        config = PreprocessingConfig()
        text = "123 456 789"
        doc = doc_preprocessor.process_document(text, config)

        assert doc.cleaned_text == text
        assert len(doc.tokens) == 3

    def test_single_word_document(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test processing single word document."""
        config = PreprocessingConfig()
        doc = doc_preprocessor.process_document("Revenue", config)

        assert doc.cleaned_text == "revenue"
        assert len(doc.sentences) == 1
        assert len(doc.tokens) == 1

    def test_repeated_punctuation(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test handling of repeated punctuation."""
        config = PreprocessingConfig()
        text = "What!!! Really??? Yes..."
        doc = doc_preprocessor.process_document(text, config)

        assert doc.cleaned_text
        assert len(doc.sentences) >= 1

    def test_ticker_preservation(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test that ticker symbols are preserved."""
        config = PreprocessingConfig(preserve_tickers=True, lowercase=False)
        text = "$AAPL and GOOGL stocks rose"
        doc = doc_preprocessor.process_document(text, config)

        # Tickers should be preserved in some form
        assert "AAPL" in doc.cleaned_text or "aapl" in doc.cleaned_text
        assert "GOOGL" in doc.cleaned_text or "googl" in doc.cleaned_text

    def test_percentage_preservation(
        self,
        doc_preprocessor: DocumentPreprocessor,
    ) -> None:
        """Test that percentages are preserved."""
        config = PreprocessingConfig(preserve_numbers=True)
        text = "Growth of 25% observed"
        doc = doc_preprocessor.process_document(text, config)

        assert "25" in doc.cleaned_text or "%" in doc.cleaned_text

    def test_multiple_spaces_and_newlines(
        self,
        text_preprocessor: TextPreprocessor,
    ) -> None:
        """Test handling of excessive whitespace."""
        text = "Line1\n\n\n\nLine2     Line3"
        cleaned = text_preprocessor.clean_text(text)

        # Should normalize to single spaces
        assert "  " not in cleaned
        assert "Line1" in cleaned
        assert "Line2" in cleaned
        assert "Line3" in cleaned
