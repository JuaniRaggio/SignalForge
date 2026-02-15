"""Base class for sector-specific analysis."""

import re
from abc import ABC, abstractmethod
from typing import TypeVar

from signalforge.nlp.sectors.schemas import SectorSignal, SignalStrength

# TypeVar for generic signal type
T_SectorSignal = TypeVar("T_SectorSignal", bound=SectorSignal)


class BaseSectorAnalyzer(ABC):
    """Base class for sector-specific analysis."""

    sector_name: str
    keywords: list[str]

    @abstractmethod
    def analyze(self, text: str, symbols: list[str]) -> list[SectorSignal]:
        """Extract sector-specific signals from text.

        Args:
            text: Input text to analyze
            symbols: List of stock symbols mentioned in the text

        Returns:
            List of extracted sector signals
        """
        ...

    @abstractmethod
    def get_sector_keywords(self) -> list[str]:
        """Return keywords that indicate this sector.

        Returns:
            List of sector-specific keywords
        """
        ...

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities relevant to sector.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        entities: dict[str, list[str]] = {
            "companies": [],
            "products": [],
            "people": [],
            "regulations": [],
        }

        # Extract company mentions (basic pattern matching)
        company_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Co|Ltd|LLC|AG)\b"
        entities["companies"] = list(set(re.findall(company_pattern, text)))

        # Extract quoted product names
        product_pattern = r'"([^"]+)"'
        entities["products"] = list(set(re.findall(product_pattern, text)))

        # Extract regulation mentions
        regulation_pattern = r"\b(?:Section\s+\d+|Act\s+\d+|Rule\s+\d+|[A-Z]{3,}(?:-\d+)?)\b"
        entities["regulations"] = list(set(re.findall(regulation_pattern, text)))

        return entities

    def calculate_sentiment(self, text: str, context_keywords: list[str]) -> float:
        """Calculate sentiment in context of sector keywords.

        Args:
            text: Input text to analyze
            context_keywords: Keywords to focus sentiment analysis on

        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        # Simple sentiment lexicon
        positive_words = {
            "approve", "approved", "approval", "positive", "growth", "increase",
            "strong", "better", "improve", "innovation", "success", "win",
            "bullish", "upgrade", "beat", "exceed", "outperform", "gain"
        }
        negative_words = {
            "reject", "rejected", "rejection", "negative", "decline", "decrease",
            "weak", "worse", "fail", "failure", "loss", "lose",
            "bearish", "downgrade", "miss", "underperform", "cut", "drop"
        }

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Check for negations
        negations = {"not", "no", "never", "neither", "nor", "none", "nobody", "nothing"}

        score = 0.0
        total_keywords = 0

        for i, word in enumerate(words):
            # Check if word is a context keyword
            is_context = any(kw.lower() in word for kw in context_keywords)

            # Check for negation
            is_negated = i > 0 and words[i - 1] in negations

            if word in positive_words:
                weight = 2.0 if is_context else 1.0
                score += -weight if is_negated else weight
                total_keywords += 1
            elif word in negative_words:
                weight = 2.0 if is_context else 1.0
                score += weight if is_negated else -weight
                total_keywords += 1

        # Normalize score
        if total_keywords > 0:
            normalized_score = max(-1.0, min(1.0, score / (total_keywords * 2)))
            return normalized_score

        return 0.0

    def map_to_signal_strength(self, sentiment: float) -> SignalStrength:
        """Map sentiment score to signal strength.

        Args:
            sentiment: Sentiment score between -1.0 and 1.0

        Returns:
            Signal strength enum
        """
        if sentiment >= 0.6:
            return SignalStrength.STRONG_BULLISH
        elif sentiment >= 0.2:
            return SignalStrength.BULLISH
        elif sentiment >= -0.2:
            return SignalStrength.NEUTRAL
        elif sentiment >= -0.6:
            return SignalStrength.BEARISH
        else:
            return SignalStrength.STRONG_BEARISH

    def _extract_symbols_from_text(self, text: str) -> list[str]:
        """Extract stock symbols from text.

        Args:
            text: Input text

        Returns:
            List of extracted symbols
        """
        # Match patterns like $AAPL or (NASDAQ: AAPL) or ticker: AAPL
        pattern = r'(?:\$|ticker:\s*|NASDAQ:\s*|NYSE:\s*)([A-Z]{1,5})\b'
        symbols = re.findall(pattern, text)
        return list(set(symbols))
