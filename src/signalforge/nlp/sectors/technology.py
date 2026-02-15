"""Technology sector analyzer."""

import re
from datetime import datetime

from signalforge.nlp.sectors.base import BaseSectorAnalyzer
from signalforge.nlp.sectors.schemas import TechnologySignal


class TechnologyAnalyzer(BaseSectorAnalyzer):
    """Analyzer for technology sector."""

    sector_name = "technology"

    def __init__(self) -> None:
        """Initialize technology analyzer with sector-specific keywords."""
        self.product_keywords = ["launch", "release", "unveil", "announce", "ship", "debut"]
        self.ai_keywords = [
            "AI", "artificial intelligence", "machine learning", "GPT", "LLM",
            "neural network", "deep learning", "transformer", "generative AI"
        ]
        self.chip_keywords = [
            "semiconductor", "chip", "GPU", "CPU", "processor", "fab",
            "foundry", "wafer", "node", "lithography", "TSMC", "ASML"
        ]
        self.competitive_keywords = [
            "market share", "competitor", "disruption", "innovation",
            "patent", "breakthrough", "rivalry", "dominance"
        ]
        self.keywords = (
            self.product_keywords + self.ai_keywords +
            self.chip_keywords + self.competitive_keywords
        )

    def get_sector_keywords(self) -> list[str]:
        """Return technology sector keywords.

        Returns:
            List of technology-specific keywords
        """
        return self.keywords

    def analyze(self, text: str, symbols: list[str]) -> list[TechnologySignal]:  # type: ignore[override]
        """Extract technology signals.

        Analyzes text for:
        1. Product cycle detection (launch, growth, decline)
        2. AI/ML relevance scoring
        3. Chip demand indicators
        4. Competitive landscape changes

        Args:
            text: Input text to analyze
            symbols: List of stock symbols mentioned

        Returns:
            List of technology signals
        """
        signals: list[TechnologySignal] = []

        # Detect product cycle signals
        product_phase, product_confidence = self.detect_product_cycle(text)
        if product_confidence > 0.4:
            sentiment = self.calculate_sentiment(text, self.product_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(TechnologySignal(
                sector=self.sector_name,
                signal_type="product_cycle",
                strength=strength,
                confidence=product_confidence,
                description=f"Product cycle phase: {product_phase}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                product_cycle_phase=product_phase,
            ))

        # AI relevance signal
        ai_score = self.score_ai_relevance(text)
        if ai_score > 0.3:
            sentiment = self.calculate_sentiment(text, self.ai_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(TechnologySignal(
                sector=self.sector_name,
                signal_type="ai_relevance",
                strength=strength,
                confidence=ai_score,
                description=f"AI/ML relevance detected (score: {ai_score:.2f})",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                ai_relevance=ai_score,
            ))

        # Chip demand signal
        chip_indicator = self.analyze_chip_demand(text)
        if chip_indicator:
            sentiment = self.calculate_sentiment(text, self.chip_keywords)
            strength = self.map_to_signal_strength(sentiment)
            confidence = 0.7 if sentiment != 0.0 else 0.5

            signals.append(TechnologySignal(
                sector=self.sector_name,
                signal_type="chip_demand",
                strength=strength,
                confidence=confidence,
                description=f"Chip demand indicator: {chip_indicator}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                chip_demand_indicator=chip_indicator,
            ))

        # Competitive landscape signal
        if any(kw.lower() in text.lower() for kw in self.competitive_keywords):
            sentiment = self.calculate_sentiment(text, self.competitive_keywords)
            strength = self.map_to_signal_strength(sentiment)
            confidence = 0.65

            signals.append(TechnologySignal(
                sector=self.sector_name,
                signal_type="competitive_landscape",
                strength=strength,
                confidence=confidence,
                description="Competitive landscape change detected",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
            ))

        return signals

    def detect_product_cycle(self, text: str) -> tuple[str, float]:
        """Detect product cycle phase and confidence.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (phase, confidence)
        """
        text_lower = text.lower()

        # Launch phase keywords
        launch_patterns = [
            r'\b(?:launch(?:es|ed|ing)?|release[ds]?|unveil(?:ed|ing)?|announce[ds]?|debut(?:ed|ing)?)\b',
            r'\b(?:new product|introducing|rollout)\b',
        ]

        # Growth phase keywords
        growth_patterns = [
            r'\b(?:growing|expansion|adoption|uptake)\b',
            r'\b(?:strong demand|increasing demand|sales growth|market penetration)\b',
        ]

        # Maturity phase keywords
        maturity_patterns = [
            r'\b(?:mature|established|stable)\b',
            r'\b(?:maintain(?:s|ed|ing)?|sustaining|plateau)\b',
        ]

        # Decline phase keywords
        decline_patterns = [
            r'\b(?:declining|declines?|phasing out|discontinu|obsolete)\b',
            r'\b(?:legacy|outdated|end-of-life)\b',
        ]

        phase_scores = {
            "launch": sum(1 for p in launch_patterns if re.search(p, text_lower)),
            "growth": sum(1 for p in growth_patterns if re.search(p, text_lower)),
            "maturity": sum(1 for p in maturity_patterns if re.search(p, text_lower)),
            "decline": sum(1 for p in decline_patterns if re.search(p, text_lower)),
        }

        max_phase = max(phase_scores, key=phase_scores.get)  # type: ignore
        max_score = phase_scores[max_phase]

        if max_score == 0:
            return "unknown", 0.0

        # Calculate confidence based on score
        confidence = min(1.0, max_score / 3.0 + 0.4)

        return max_phase, confidence

    def score_ai_relevance(self, text: str) -> float:
        """Score AI/ML relevance of the content.

        Args:
            text: Input text to analyze

        Returns:
            AI relevance score between 0.0 and 1.0
        """
        text_lower = text.lower()

        # Count AI keyword mentions
        ai_mentions = sum(
            1 for keyword in self.ai_keywords
            if keyword.lower() in text_lower
        )

        # Bonus for multiple AI concepts
        if ai_mentions >= 3:
            return min(1.0, 0.6 + (ai_mentions - 3) * 0.1)
        elif ai_mentions >= 2:
            return 0.5
        elif ai_mentions == 1:
            return 0.35
        else:
            return 0.0

    def analyze_chip_demand(self, text: str) -> str | None:
        """Analyze chip/semiconductor demand signals.

        Args:
            text: Input text to analyze

        Returns:
            Chip demand indicator or None
        """
        text_lower = text.lower()

        # Check for chip shortage indicators
        shortage_keywords = ["shortage", "supply constraint", "allocation", "lead time"]
        if (any(kw in text_lower for kw in shortage_keywords) and
                any(chip_kw.lower() in text_lower for chip_kw in self.chip_keywords)):
            return "shortage"

        # Check for strong demand
        demand_keywords = ["strong demand", "robust demand", "high demand", "capacity expansion"]
        if (any(kw in text_lower for kw in demand_keywords) and
                any(chip_kw.lower() in text_lower for chip_kw in self.chip_keywords)):
            return "high_demand"

        # Check for weak demand
        weak_keywords = ["weak demand", "soft demand", "inventory correction", "utilization decline"]
        if (any(kw in text_lower for kw in weak_keywords) and
                any(chip_kw.lower() in text_lower for chip_kw in self.chip_keywords)):
            return "weak_demand"

        # Check for any chip mention
        if any(chip_kw.lower() in text_lower for chip_kw in self.chip_keywords):
            return "neutral"

        return None
