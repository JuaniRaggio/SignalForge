"""Consumer sector analyzer."""

from datetime import datetime

from signalforge.nlp.sectors.base import BaseSectorAnalyzer
from signalforge.nlp.sectors.schemas import ConsumerSignal


class ConsumerAnalyzer(BaseSectorAnalyzer):
    """Analyzer for consumer sector."""

    sector_name = "consumer"

    def __init__(self) -> None:
        """Initialize consumer analyzer with sector-specific keywords."""
        self.sentiment_keywords = [
            "consumer confidence", "spending", "demand", "sales",
            "retail", "discretionary", "consumer sentiment", "purchasing power"
        ]
        self.seasonal_keywords = [
            "holiday", "back-to-school", "summer", "Q4", "Christmas",
            "Black Friday", "Cyber Monday", "seasonal", "year-end"
        ]
        self.brand_keywords = [
            "brand", "loyalty", "market share", "pricing power",
            "brand strength", "customer retention", "brand equity", "premium"
        ]
        self.competitive_keywords = [
            "competitor", "competition", "market leader", "differentiation",
            "value proposition", "competitive advantage"
        ]
        self.keywords = (
            self.sentiment_keywords + self.seasonal_keywords +
            self.brand_keywords + self.competitive_keywords
        )

    def get_sector_keywords(self) -> list[str]:
        """Return consumer sector keywords.

        Returns:
            List of consumer-specific keywords
        """
        return self.keywords

    def analyze(self, text: str, symbols: list[str]) -> list[ConsumerSignal]:  # type: ignore[override]
        """Extract consumer signals.

        Analyzes text for:
        1. Consumer sentiment indicators
        2. Seasonal factors
        3. Brand momentum
        4. Pricing power signals

        Args:
            text: Input text to analyze
            symbols: List of stock symbols mentioned

        Returns:
            List of consumer signals
        """
        signals: list[ConsumerSignal] = []

        # Detect consumer sentiment
        consumer_sent = self.analyze_consumer_sentiment(text)
        if consumer_sent is not None:
            strength = self.map_to_signal_strength(consumer_sent)

            signals.append(ConsumerSignal(
                sector=self.sector_name,
                signal_type="consumer_sentiment",
                strength=strength,
                confidence=0.7,
                description=f"Consumer sentiment score: {consumer_sent:.2f}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                consumer_sentiment=consumer_sent,
            ))

        # Detect seasonal factors
        seasonal_factor = self.detect_seasonal_factor(text)
        if seasonal_factor:
            sentiment = self.calculate_sentiment(text, self.seasonal_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(ConsumerSignal(
                sector=self.sector_name,
                signal_type="seasonal",
                strength=strength,
                confidence=0.65,
                description=f"Seasonal factor: {seasonal_factor}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                seasonal_factor=seasonal_factor,
            ))

        # Detect brand momentum
        brand_momentum = self.analyze_brand_momentum(text)
        if brand_momentum:
            sentiment = self.calculate_sentiment(text, self.brand_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(ConsumerSignal(
                sector=self.sector_name,
                signal_type="brand_momentum",
                strength=strength,
                confidence=0.7,
                description=f"Brand momentum: {brand_momentum}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                brand_momentum=brand_momentum,
            ))

        # Detect pricing power signals
        if self.detect_pricing_power(text):
            sentiment = self.calculate_sentiment(text, ["pricing", "price", "margin"])
            strength = self.map_to_signal_strength(sentiment)

            signals.append(ConsumerSignal(
                sector=self.sector_name,
                signal_type="pricing_power",
                strength=strength,
                confidence=0.65,
                description="Pricing power signal detected",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
            ))

        return signals

    def analyze_consumer_sentiment(self, text: str) -> float | None:
        """Analyze consumer sentiment from text.

        Args:
            text: Input text to analyze

        Returns:
            Consumer sentiment score between -1.0 and 1.0, or None
        """
        text_lower = text.lower()

        # Check if sentiment-related content is present
        sentiment_present = any(kw.lower() in text_lower for kw in self.sentiment_keywords)
        if not sentiment_present:
            return None

        # Positive sentiment indicators
        positive_indicators = [
            "strong demand", "robust sales", "increased spending",
            "consumer confidence rising", "high demand", "sales growth",
            "spending increase", "confident consumer"
        ]

        # Negative sentiment indicators
        negative_indicators = [
            "weak demand", "soft sales", "decreased spending",
            "consumer confidence falling", "low demand", "sales decline",
            "spending decrease", "cautious consumer"
        ]

        positive_count = sum(1 for ind in positive_indicators if ind in text_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in text_lower)

        if positive_count == 0 and negative_count == 0:
            # Use general sentiment analysis
            return self.calculate_sentiment(text, self.sentiment_keywords)

        # Calculate net sentiment
        net_sentiment = (positive_count - negative_count) / max(1, positive_count + negative_count)

        return max(-1.0, min(1.0, net_sentiment))

    def detect_seasonal_factor(self, text: str) -> str | None:
        """Detect seasonal factors in consumer behavior.

        Args:
            text: Input text to analyze

        Returns:
            Seasonal factor identifier or None
        """
        text_lower = text.lower()

        # Holiday season (Q4)
        if any(kw in text_lower for kw in ["holiday", "christmas", "q4", "year-end", "festive"]):
            return "holiday_season"

        # Black Friday / Cyber Monday
        if any(kw in text_lower for kw in ["black friday", "cyber monday"]):
            return "black_friday"

        # Back to school
        if "back-to-school" in text_lower or "back to school" in text_lower:
            return "back_to_school"

        # Summer season
        if any(kw in text_lower for kw in ["summer", "vacation season"]):
            return "summer"

        # General seasonal mention
        if "seasonal" in text_lower:
            return "seasonal"

        return None

    def analyze_brand_momentum(self, text: str) -> str | None:
        """Analyze brand momentum indicators.

        Args:
            text: Input text to analyze

        Returns:
            Brand momentum indicator or None
        """
        text_lower = text.lower()

        # Check for brand-related content
        brand_present = any(kw.lower() in text_lower for kw in self.brand_keywords)
        if not brand_present:
            return None

        # Negative momentum (check first to avoid false positives)
        if any(kw in text_lower for kw in [
            "brand erosion", "losing market share", "declining loyalty",
            "brand weakness", "commoditization"
        ]):
            return "negative"

        # Strong positive momentum
        if any(kw in text_lower for kw in [
            "brand strength", "increasing loyalty", "growing market share",
            "brand leader", "premium brand", "brand equity increase"
        ]):
            return "strong_positive"

        # Positive momentum
        if any(kw in text_lower for kw in [
            "brand improvement", "loyalty", "market share gain"
        ]):
            return "positive"

        # Stable brand
        if any(kw in text_lower for kw in ["stable", "maintain", "consistent"]):
            return "stable"

        return "neutral"

    def detect_pricing_power(self, text: str) -> bool:
        """Detect pricing power signals.

        Args:
            text: Input text to analyze

        Returns:
            True if pricing power signal detected
        """
        text_lower = text.lower()

        pricing_keywords = [
            "pricing power", "price increase", "margin expansion",
            "premium pricing", "pass through costs", "maintain prices",
            "pricing flexibility", "price hike"
        ]

        return any(kw in text_lower for kw in pricing_keywords)
