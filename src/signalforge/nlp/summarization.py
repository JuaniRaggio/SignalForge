"""Text summarization for financial content.

This module provides extractive summarization capabilities for financial documents,
news articles, and earnings transcripts. It uses heuristic-based approaches including
TF-IDF scoring, sentence position, and financial term weighting.

Key Features:
- Single document summarization with configurable length and style
- Multi-document summarization with theme detection
- Key point extraction
- Entity extraction (tickers, companies, people)
- Headline generation
- Specialized earnings call summarization
- Executive summary generation

Examples:
    Basic text summarization:

    >>> from signalforge.nlp.summarization import TextSummarizer, SummaryLength, SummaryStyle
    >>>
    >>> summarizer = TextSummarizer()
    >>> text = "Apple reported strong Q4 earnings..."
    >>> result = summarizer.summarize(text, length=SummaryLength.SHORT, style=SummaryStyle.NEUTRAL)
    >>> print(result.summary)

    Multi-document summarization:

    >>> texts = ["Article 1 text...", "Article 2 text...", "Article 3 text..."]
    >>> multi_summary = summarizer.summarize_multiple(texts)
    >>> print(multi_summary.combined_summary)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Any

from signalforge.core.logging import get_logger
from signalforge.nlp.preprocessing import TextPreprocessor

logger = get_logger(__name__)


class SummaryLength(str, Enum):
    """Summary length options."""

    BRIEF = "brief"  # 1-2 sentences
    SHORT = "short"  # 3-4 sentences
    MEDIUM = "medium"  # 5-7 sentences
    DETAILED = "detailed"  # 8-10 sentences


class SummaryStyle(str, Enum):
    """Summary style options."""

    NEUTRAL = "neutral"  # Factual, no opinion
    ACTIONABLE = "actionable"  # Focus on what to do
    ANALYTICAL = "analytical"  # Include analysis
    EXECUTIVE = "executive"  # High-level overview


@dataclass
class SummaryResult:
    """Result of text summarization.

    Attributes:
        summary: Generated summary text.
        length: Length category of the summary.
        style: Style used for summarization.
        key_points: List of extracted key points.
        entities_mentioned: List of tickers, companies, and people mentioned.
        sentiment_summary: Brief sentiment indication.
        word_count: Number of words in the summary.
        compression_ratio: Ratio of original to summary length.
    """

    summary: str
    length: SummaryLength
    style: SummaryStyle
    key_points: list[str]
    entities_mentioned: list[str]
    sentiment_summary: str | None
    word_count: int
    compression_ratio: float

    def __post_init__(self) -> None:
        """Validate summary result fields."""
        if self.compression_ratio < 0:
            raise ValueError(f"Compression ratio must be non-negative, got {self.compression_ratio}")


@dataclass
class MultiDocumentSummary:
    """Result of multi-document summarization.

    Attributes:
        combined_summary: Overall summary across all documents.
        document_count: Number of documents summarized.
        common_themes: List of common themes across documents.
        conflicting_points: List of conflicting information found.
        timeline: Chronological timeline of events.
        key_entities: Most frequently mentioned entities.
    """

    combined_summary: str
    document_count: int
    common_themes: list[str]
    conflicting_points: list[str]
    timeline: list[tuple[str, str]]
    key_entities: list[str]

    def __post_init__(self) -> None:
        """Validate multi-document summary fields."""
        if self.document_count < 0:
            raise ValueError(f"Document count must be non-negative, got {self.document_count}")


class TextSummarizer:
    """Summarize financial text content using extractive methods.

    This class implements extractive summarization using TF-IDF scoring,
    sentence position weighting, and financial term importance.

    Examples:
        >>> summarizer = TextSummarizer()
        >>> result = summarizer.summarize(text, SummaryLength.SHORT)
        >>> print(result.summary)
    """

    # Sentence importance indicators (financial-specific)
    IMPORTANT_PHRASES = [
        "announced",
        "reported",
        "expects",
        "guidance",
        "beat",
        "miss",
        "raised",
        "lowered",
        "upgraded",
        "downgraded",
        "initiated",
        "target",
        "rating",
        "revenue",
        "earnings",
        "profit",
        "loss",
        "growth",
        "decline",
        "increase",
        "decrease",
        "forecast",
        "outlook",
        "dividend",
        "buyback",
        "acquisition",
        "merger",
        "CEO",
        "CFO",
        "management",
    ]

    # Length mappings
    LENGTH_MAP = {
        SummaryLength.BRIEF: (1, 2),
        SummaryLength.SHORT: (3, 4),
        SummaryLength.MEDIUM: (5, 7),
        SummaryLength.DETAILED: (8, 10),
    }

    def __init__(self) -> None:
        """Initialize the text summarizer."""
        self._preprocessor = TextPreprocessor()
        self._ticker_pattern = re.compile(r"\$?[A-Z]{1,5}\b")
        self._company_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Inc\.?|\s+Corp\.?|\s+LLC)?\b")
        self._person_pattern = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
        logger.info("text_summarizer_initialized")

    def summarize(
        self,
        text: str,
        length: SummaryLength = SummaryLength.SHORT,
        style: SummaryStyle = SummaryStyle.NEUTRAL,
        max_sentences: int | None = None,
    ) -> SummaryResult:
        """Summarize a single document.

        Args:
            text: Input text to summarize.
            length: Desired summary length.
            style: Desired summary style.
            max_sentences: Optional override for maximum sentences.

        Returns:
            SummaryResult containing the summary and metadata.

        Raises:
            ValueError: If text is empty.

        Examples:
            >>> summarizer = TextSummarizer()
            >>> result = summarizer.summarize(text, SummaryLength.SHORT)
            >>> print(result.summary)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.info("summarizing_text", text_length=len(text), length=length, style=style)

        # Clean and extract sentences
        cleaned_text = self._preprocessor.clean_text(text)
        sentences = self._preprocessor.extract_sentences(cleaned_text)

        if not sentences:
            raise ValueError("No valid sentences found in text")

        # Determine target sentence count
        if max_sentences:
            target_count = max_sentences
        else:
            min_count, max_count = self.LENGTH_MAP[length]
            target_count = min(max_count, len(sentences))
            target_count = max(min_count, target_count)

        # Score and select sentences
        scored_sentences = self._score_sentences(sentences, cleaned_text, style)
        selected_sentences = self._select_top_sentences(scored_sentences, target_count)

        # Generate summary
        summary = " ".join(selected_sentences)

        # Extract key points
        key_points = self.extract_key_points(text, max_points=5)

        # Extract entities
        entities = self.extract_entities(text)

        # Calculate metrics
        word_count = len(summary.split())
        original_word_count = len(text.split())
        compression_ratio = original_word_count / word_count if word_count > 0 else 0.0

        # Generate sentiment summary (placeholder)
        sentiment_summary = self._generate_sentiment_summary(text)

        result = SummaryResult(
            summary=summary,
            length=length,
            style=style,
            key_points=key_points,
            entities_mentioned=entities,
            sentiment_summary=sentiment_summary,
            word_count=word_count,
            compression_ratio=compression_ratio,
        )

        logger.info(
            "summarization_complete",
            word_count=word_count,
            compression_ratio=compression_ratio,
            num_sentences=len(selected_sentences),
        )

        return result

    def summarize_multiple(
        self,
        texts: list[str],
        length: SummaryLength = SummaryLength.MEDIUM,
    ) -> MultiDocumentSummary:
        """Summarize multiple documents together.

        Args:
            texts: List of text documents to summarize.
            length: Desired summary length.

        Returns:
            MultiDocumentSummary with combined analysis.

        Raises:
            ValueError: If texts list is empty.

        Examples:
            >>> summarizer = TextSummarizer()
            >>> texts = ["Doc 1...", "Doc 2...", "Doc 3..."]
            >>> result = summarizer.summarize_multiple(texts)
            >>> print(result.combined_summary)
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.info("summarizing_multiple_documents", num_documents=len(texts))

        # Combine all texts
        combined_text = "\n\n".join(texts)

        # Generate combined summary
        summary_result = self.summarize(combined_text, length=length)

        # Detect common themes
        common_themes = self.detect_common_themes(texts)

        # Extract entities from all documents
        all_entities: list[str] = []
        for text in texts:
            all_entities.extend(self.extract_entities(text))

        # Count and get top entities
        entity_counts = Counter(all_entities)
        key_entities = [entity for entity, _ in entity_counts.most_common(10)]

        # Detect conflicting points (placeholder for MVP)
        conflicting_points = self._detect_conflicts(texts)

        # Build timeline (placeholder for MVP)
        timeline = self._extract_timeline(texts)

        result = MultiDocumentSummary(
            combined_summary=summary_result.summary,
            document_count=len(texts),
            common_themes=common_themes,
            conflicting_points=conflicting_points,
            timeline=timeline,
            key_entities=key_entities,
        )

        logger.info("multi_document_summarization_complete", num_themes=len(common_themes))

        return result

    def extract_key_points(
        self,
        text: str,
        max_points: int = 5,
    ) -> list[str]:
        """Extract key points as bullet points.

        Args:
            text: Input text.
            max_points: Maximum number of key points to extract.

        Returns:
            List of key point strings.

        Examples:
            >>> summarizer = TextSummarizer()
            >>> points = summarizer.extract_key_points(text, max_points=3)
        """
        if not text or not text.strip():
            return []

        sentences = self._preprocessor.extract_sentences(text)

        if not sentences:
            return []

        # Score sentences for importance
        scored = self._score_sentences(sentences, text, SummaryStyle.NEUTRAL)

        # Select top sentences as key points
        key_points = self._select_top_sentences(scored, min(max_points, len(scored)))

        logger.debug("key_points_extracted", count=len(key_points))

        return key_points

    def extract_entities(
        self,
        text: str,
    ) -> list[str]:
        """Extract mentioned entities (tickers, companies, people).

        Args:
            text: Input text.

        Returns:
            List of entity strings.

        Examples:
            >>> summarizer = TextSummarizer()
            >>> entities = summarizer.extract_entities(text)
        """
        if not text:
            return []

        entities: list[str] = []

        # Extract tickers
        tickers = self._ticker_pattern.findall(text)
        entities.extend(tickers)

        # Extract company names
        companies = self._company_pattern.findall(text)
        entities.extend(companies)

        # Extract person names
        persons = self._person_pattern.findall(text)
        entities.extend(persons)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_clean = entity.strip()
            if entity_clean and entity_clean not in seen:
                seen.add(entity_clean)
                unique_entities.append(entity_clean)

        logger.debug("entities_extracted", count=len(unique_entities))

        return unique_entities

    def score_sentence_importance(
        self,
        sentence: str,
        full_text: str,
    ) -> float:
        """Score how important a sentence is.

        Args:
            sentence: Sentence to score.
            full_text: Full text context.

        Returns:
            Importance score (0-1).

        Examples:
            >>> summarizer = TextSummarizer()
            >>> score = summarizer.score_sentence_importance(sentence, full_text)
        """
        if not sentence or not full_text:
            return 0.0

        score = 0.0

        # Check for important financial phrases
        sentence_lower = sentence.lower()
        for phrase in self.IMPORTANT_PHRASES:
            if phrase in sentence_lower:
                score += 0.1

        # Check for numeric values
        if re.search(r"\d+(?:\.\d+)?%?", sentence):
            score += 0.15

        # Check for entities
        if self._ticker_pattern.search(sentence):
            score += 0.2

        # TF-IDF component (simplified)
        words = sentence.lower().split()
        full_words = full_text.lower().split()

        if not full_words:
            return min(score, 1.0)

        tf_idf_score = 0.0
        for word in words:
            if len(word) > 3:  # Skip short words
                tf = full_words.count(word) / len(full_words)
                # Simple IDF approximation
                idf = log(len(full_words) / (full_words.count(word) + 1))
                tf_idf_score += tf * idf

        score += min(tf_idf_score / len(words) if words else 0.0, 0.3)

        # Normalize to 0-1
        return min(score, 1.0)

    def generate_headline(
        self,
        text: str,
        max_words: int = 10,
    ) -> str:
        """Generate a headline from the text.

        Args:
            text: Input text.
            max_words: Maximum words in headline.

        Returns:
            Generated headline string.

        Examples:
            >>> summarizer = TextSummarizer()
            >>> headline = summarizer.generate_headline(text, max_words=8)
        """
        if not text or not text.strip():
            return ""

        sentences = self._preprocessor.extract_sentences(text)

        if not sentences:
            return ""

        # Get the highest scoring sentence
        scored = self._score_sentences(sentences, text, SummaryStyle.NEUTRAL)

        if not scored:
            return ""

        top_sentence = scored[0][1]

        # Truncate to max words
        words = top_sentence.split()
        headline_words = words[:max_words]
        headline = " ".join(headline_words)

        if len(words) > max_words:
            headline += "..."

        logger.debug("headline_generated", headline=headline)

        return headline

    def create_executive_summary(
        self,
        text: str,
        symbol: str | None = None,
    ) -> str:
        """Create an executive-style summary.

        Args:
            text: Input text.
            symbol: Optional stock symbol for context.

        Returns:
            Executive summary string.

        Examples:
            >>> summarizer = TextSummarizer()
            >>> exec_summary = summarizer.create_executive_summary(text, "AAPL")
        """
        result = self.summarize(text, length=SummaryLength.SHORT, style=SummaryStyle.EXECUTIVE)

        lines: list[str] = []

        if symbol:
            lines.append(f"Executive Summary: {symbol}")
            lines.append("=" * 50)
        else:
            lines.append("Executive Summary")
            lines.append("=" * 50)

        lines.append("")
        lines.append(result.summary)
        lines.append("")

        if result.key_points:
            lines.append("Key Points:")
            for i, point in enumerate(result.key_points[:3], 1):
                lines.append(f"{i}. {point}")

        if result.entities_mentioned:
            lines.append("")
            lines.append(f"Mentioned: {', '.join(result.entities_mentioned[:5])}")

        executive_summary = "\n".join(lines)
        logger.debug("executive_summary_created", length=len(executive_summary))

        return executive_summary

    def summarize_earnings_call(
        self,
        transcript: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Specialized summarization for earnings calls.

        Args:
            transcript: Full earnings call transcript.
            symbol: Stock symbol.

        Returns:
            Dictionary with guidance (str), key_metrics (list[str]),
            management_tone (str), and qa_highlights (list[str]).

        Examples:
            >>> summarizer = TextSummarizer()
            >>> summary = summarizer.summarize_earnings_call(transcript, "AAPL")
            >>> print(summary['guidance'])
        """
        if not transcript or not transcript.strip():
            return {
                "guidance": "",
                "key_metrics": [],
                "management_tone": "neutral",
                "qa_highlights": [],
            }

        logger.info("summarizing_earnings_call", symbol=symbol)

        # Detect sections (prepared remarks vs Q&A)
        sections = self._split_earnings_sections(transcript)

        # Extract guidance
        guidance = self._extract_guidance(sections.get("prepared_remarks", ""))

        # Extract key metrics
        key_metrics = self._extract_key_metrics(transcript)

        # Determine management tone
        management_tone = self._assess_management_tone(sections.get("prepared_remarks", ""))

        # Extract Q&A highlights
        qa_highlights = self._extract_qa_highlights(sections.get("qa", ""))

        result = {
            "guidance": guidance,
            "key_metrics": key_metrics,
            "management_tone": management_tone,
            "qa_highlights": qa_highlights,
        }

        logger.info("earnings_call_summarized", num_metrics=len(key_metrics))

        return result

    def detect_common_themes(
        self,
        texts: list[str],
    ) -> list[str]:
        """Detect common themes across multiple texts.

        Args:
            texts: List of text documents.

        Returns:
            List of common theme strings.

        Examples:
            >>> summarizer = TextSummarizer()
            >>> themes = summarizer.detect_common_themes(texts)
        """
        if not texts:
            return []

        # Extract important words from all texts
        all_words: list[str] = []

        for text in texts:
            words = self._preprocessor.tokenize(text.lower())
            # Filter to meaningful words
            meaningful_words = [
                w for w in words if len(w) > 4 and w not in {"their", "which", "would", "could", "should"}
            ]
            all_words.extend(meaningful_words)

        # Find most common words/phrases
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(10) if count >= len(texts) // 2]

        logger.debug("common_themes_detected", count=len(common_words))

        return common_words

    def _score_sentences(
        self,
        sentences: list[str],
        full_text: str,
        style: SummaryStyle,
    ) -> list[tuple[float, str]]:
        """Score sentences for importance.

        Args:
            sentences: List of sentences to score.
            full_text: Full text for context.
            style: Summary style to apply.

        Returns:
            List of tuples (score, sentence) sorted by score descending.
        """
        scored: list[tuple[float, str]] = []

        for i, sentence in enumerate(sentences):
            base_score = self.score_sentence_importance(sentence, full_text)

            # Position bonus (first and last sentences are often important)
            position_bonus = 0.0
            if i == 0:
                position_bonus = 0.2
            elif i == len(sentences) - 1:
                position_bonus = 0.1

            # Style-specific adjustments
            style_bonus = 0.0
            sentence_lower = sentence.lower()

            if style == SummaryStyle.ACTIONABLE:
                if any(word in sentence_lower for word in ["should", "must", "recommend", "expect", "will"]):
                    style_bonus = 0.15

            elif style == SummaryStyle.ANALYTICAL:
                if any(word in sentence_lower for word in ["because", "due to", "resulted", "indicates", "suggests"]):
                    style_bonus = 0.15

            elif style == SummaryStyle.EXECUTIVE and len(sentence.split()) < 20:
                # Prefer concise, high-level statements
                style_bonus = 0.1

            final_score = base_score + position_bonus + style_bonus
            scored.append((final_score, sentence))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored

    def _select_top_sentences(
        self,
        scored_sentences: list[tuple[float, str]],
        count: int,
    ) -> list[str]:
        """Select top N sentences and return in original order.

        Args:
            scored_sentences: List of (score, sentence) tuples.
            count: Number of sentences to select.

        Returns:
            List of selected sentences in original order.
        """
        # Take top N by score
        top_scored = scored_sentences[:count]

        # Return in original order (by position in text)
        # For simplicity, return in score order for now
        return [sentence for _, sentence in top_scored]

    def _generate_sentiment_summary(self, text: str) -> str | None:
        """Generate brief sentiment indication.

        Args:
            text: Input text.

        Returns:
            Sentiment summary string or None.
        """
        # Simple keyword-based sentiment (placeholder)
        text_lower = text.lower()

        positive_words = ["beat", "exceeded", "strong", "growth", "positive", "gained", "up", "raised"]
        negative_words = ["miss", "weak", "decline", "loss", "negative", "fell", "down", "lowered"]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count + 2:
            return "positive"
        elif neg_count > pos_count + 2:
            return "negative"
        else:
            return "neutral"

    def _detect_conflicts(self, texts: list[str]) -> list[str]:
        """Detect conflicting information across texts.

        Args:
            texts: List of text documents.

        Returns:
            List of conflict descriptions.
        """
        # Placeholder for MVP
        conflicts: list[str] = []

        # Simple heuristic: look for contradictory keywords
        has_positive = any("exceeded" in t.lower() or "beat" in t.lower() for t in texts)
        has_negative = any("missed" in t.lower() or "fell short" in t.lower() for t in texts)

        if has_positive and has_negative:
            conflicts.append("Conflicting reports on performance vs expectations")

        return conflicts

    def _extract_timeline(self, texts: list[str]) -> list[tuple[str, str]]:
        """Extract timeline of events from texts.

        Args:
            texts: List of text documents.

        Returns:
            List of (date, event) tuples.
        """
        # Placeholder for MVP
        timeline: list[tuple[str, str]] = []

        # Extract date patterns and associated sentences
        date_pattern = re.compile(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2}, \d{4})\b")

        for text in texts:
            sentences = self._preprocessor.extract_sentences(text)
            for sentence in sentences:
                dates = date_pattern.findall(sentence)
                if dates:
                    # Take first 50 chars of sentence as event
                    event = sentence[:50] + "..." if len(sentence) > 50 else sentence
                    timeline.append((dates[0], event))

        return timeline[:10]  # Return top 10

    def _split_earnings_sections(self, transcript: str) -> dict[str, str]:
        """Split earnings call into sections.

        Args:
            transcript: Full transcript text.

        Returns:
            Dictionary with section names and content.
        """
        sections: dict[str, str] = {}

        # Simple heuristic: split on Q&A marker
        qa_markers = [
            "question-and-answer",
            "q&a",
            "questions and answers",
            "operator",
            "first question",
        ]

        transcript_lower = transcript.lower()

        # Find Q&A start
        qa_start = -1
        for marker in qa_markers:
            pos = transcript_lower.find(marker)
            if pos != -1:
                qa_start = pos
                break

        if qa_start != -1:
            sections["prepared_remarks"] = transcript[:qa_start]
            sections["qa"] = transcript[qa_start:]
        else:
            sections["prepared_remarks"] = transcript
            sections["qa"] = ""

        return sections

    def _extract_guidance(self, text: str) -> str:
        """Extract guidance from earnings text.

        Args:
            text: Earnings text.

        Returns:
            Guidance summary string.
        """
        if not text:
            return ""

        # Look for guidance keywords
        guidance_keywords = ["guidance", "outlook", "expect", "forecast", "anticipate", "project"]

        sentences = self._preprocessor.extract_sentences(text)

        guidance_sentences: list[str] = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in guidance_keywords):
                guidance_sentences.append(sentence)

        if guidance_sentences:
            return " ".join(guidance_sentences[:2])  # Top 2 guidance sentences

        return "No specific guidance provided"

    def _extract_key_metrics(self, text: str) -> list[str]:
        """Extract key financial metrics.

        Args:
            text: Earnings text.

        Returns:
            List of metric strings.
        """
        metrics: list[str] = []

        # Look for metric keywords
        metric_keywords = ["revenue", "earnings", "eps", "profit", "margin", "growth", "sales"]

        sentences = self._preprocessor.extract_sentences(text)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains metrics and numbers
            if any(kw in sentence_lower for kw in metric_keywords) and re.search(r"\d+", sentence):
                metrics.append(sentence)

        return metrics[:5]  # Top 5 metric sentences

    def _assess_management_tone(self, text: str) -> str:
        """Assess management tone from prepared remarks.

        Args:
            text: Prepared remarks text.

        Returns:
            Tone assessment (positive, neutral, negative).
        """
        if not text:
            return "neutral"

        sentiment = self._generate_sentiment_summary(text)
        return sentiment or "neutral"

    def _extract_qa_highlights(self, qa_text: str) -> list[str]:
        """Extract Q&A highlights.

        Args:
            qa_text: Q&A section text.

        Returns:
            List of highlight strings.
        """
        if not qa_text:
            return []

        sentences = self._preprocessor.extract_sentences(qa_text)

        # Score sentences and take top ones
        if not sentences:
            return []

        scored = self._score_sentences(sentences, qa_text, SummaryStyle.NEUTRAL)
        highlights = self._select_top_sentences(scored, min(3, len(scored)))

        return highlights
