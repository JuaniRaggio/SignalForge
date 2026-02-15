"""Financial sector analyzer."""

import re
from datetime import datetime

from signalforge.nlp.sectors.base import BaseSectorAnalyzer
from signalforge.nlp.sectors.schemas import FinancialSignal


class FinancialAnalyzer(BaseSectorAnalyzer):
    """Analyzer for financial sector."""

    sector_name = "financial"

    def __init__(self) -> None:
        """Initialize financial analyzer with sector-specific keywords."""
        self.rate_keywords = [
            "interest rate", "Fed", "FOMC", "rate hike", "rate cut", "basis points",
            "bps", "federal reserve", "monetary policy", "yield curve", "fed funds"
        ]
        self.credit_keywords = [
            "credit quality", "default", "NPL", "provision", "write-off",
            "non-performing loan", "delinquency", "credit loss", "charge-off",
            "loan loss reserve"
        ]
        self.regulatory_keywords = [
            "Basel", "capital requirement", "stress test", "compliance",
            "Dodd-Frank", "capital ratio", "tier 1", "CCAR", "regulatory capital"
        ]
        self.loan_keywords = [
            "loan growth", "lending", "mortgage", "commercial loan",
            "consumer loan", "loan portfolio", "credit expansion"
        ]
        self.keywords = (
            self.rate_keywords + self.credit_keywords +
            self.regulatory_keywords + self.loan_keywords
        )

    def get_sector_keywords(self) -> list[str]:
        """Return financial sector keywords.

        Returns:
            List of financial-specific keywords
        """
        return self.keywords

    def analyze(self, text: str, symbols: list[str]) -> list[FinancialSignal]:  # type: ignore[override]
        """Extract financial sector signals.

        Analyzes text for:
        1. Interest rate sensitivity
        2. Credit quality indicators
        3. Regulatory impact
        4. Loan growth signals

        Args:
            text: Input text to analyze
            symbols: List of stock symbols mentioned

        Returns:
            List of financial signals
        """
        signals: list[FinancialSignal] = []

        # Detect interest rate sensitivity
        rate_sensitivity = self.analyze_rate_sensitivity(text)
        if rate_sensitivity is not None:
            sentiment = self.calculate_sentiment(text, self.rate_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(FinancialSignal(
                sector=self.sector_name,
                signal_type="interest_rate",
                strength=strength,
                confidence=0.75,
                description=f"Interest rate sensitivity: {rate_sensitivity:.2f}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                interest_rate_sensitivity=rate_sensitivity,
            ))

        # Detect credit quality signals
        credit_indicator = self.analyze_credit_quality(text)
        if credit_indicator:
            sentiment = self.calculate_sentiment(text, self.credit_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(FinancialSignal(
                sector=self.sector_name,
                signal_type="credit_quality",
                strength=strength,
                confidence=0.7,
                description=f"Credit quality indicator: {credit_indicator}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                credit_quality_indicator=credit_indicator,
            ))

        # Detect regulatory impact
        regulatory_impact = self.analyze_regulatory_impact(text)
        if regulatory_impact:
            sentiment = self.calculate_sentiment(text, self.regulatory_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(FinancialSignal(
                sector=self.sector_name,
                signal_type="regulatory",
                strength=strength,
                confidence=0.65,
                description=f"Regulatory impact: {regulatory_impact}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                regulatory_impact=regulatory_impact,
            ))

        # Detect loan growth signals
        if any(kw.lower() in text.lower() for kw in self.loan_keywords):
            sentiment = self.calculate_sentiment(text, self.loan_keywords)
            if abs(sentiment) > 0.2:  # Only create signal if meaningful
                strength = self.map_to_signal_strength(sentiment)
                confidence = 0.6

                signals.append(FinancialSignal(
                    sector=self.sector_name,
                    signal_type="loan_growth",
                    strength=strength,
                    confidence=confidence,
                    description="Loan growth signal detected",
                    affected_symbols=symbols or self._extract_symbols_from_text(text),
                    source_text=text[:500],
                    timestamp=datetime.utcnow(),
                ))

        return signals

    def analyze_rate_sensitivity(self, text: str) -> float | None:
        """Analyze interest rate sensitivity.

        Args:
            text: Input text to analyze

        Returns:
            Rate sensitivity score between 0.0 and 1.0, or None
        """
        text_lower = text.lower()

        # Check if rate-related content is present
        rate_present = any(kw.lower() in text_lower for kw in self.rate_keywords)
        if not rate_present:
            return None

        # High sensitivity indicators
        high_sensitivity_keywords = [
            "rate hike", "rate cut", "50 basis points", "75 basis points",
            "100 basis points", "aggressive", "fed meeting", "FOMC decision"
        ]

        # Extract basis points if mentioned
        bps_pattern = r'(\d+)\s*(?:basis points|bps)'
        bps_matches = re.findall(bps_pattern, text_lower)
        if bps_matches:
            max_bps = max(int(bps) for bps in bps_matches)
            # Higher basis points = higher sensitivity
            sensitivity = min(1.0, max_bps / 100.0)
            return sensitivity

        # Check for high sensitivity keywords
        high_count = sum(1 for kw in high_sensitivity_keywords if kw in text_lower)
        if high_count > 0:
            return min(1.0, 0.6 + (high_count * 0.15))

        # Default moderate sensitivity if rates mentioned
        return 0.4

    def analyze_credit_quality(self, text: str) -> str | None:
        """Analyze credit quality indicators.

        Args:
            text: Input text to analyze

        Returns:
            Credit quality indicator or None
        """
        text_lower = text.lower()

        # Check for credit-related content
        credit_present = any(kw.lower() in text_lower for kw in self.credit_keywords)
        if not credit_present:
            return None

        # Deteriorating credit quality
        if any(kw in text_lower for kw in [
            "rising defaults", "increasing NPL", "credit deterioration",
            "write-off", "charge-off", "provision increase"
        ]):
            return "deteriorating"

        # Improving credit quality
        if any(kw in text_lower for kw in [
            "declining defaults", "decreasing NPL", "credit improvement",
            "lower provisions", "credit quality improvement"
        ]):
            return "improving"

        # Stable credit quality
        if any(kw in text_lower for kw in ["stable", "maintain", "consistent"]):
            return "stable"

        # General credit mention
        return "neutral"

    def analyze_regulatory_impact(self, text: str) -> str | None:
        """Analyze regulatory impact signals.

        Args:
            text: Input text to analyze

        Returns:
            Regulatory impact indicator or None
        """
        text_lower = text.lower()

        # Check for regulatory content
        regulatory_present = any(kw.lower() in text_lower for kw in self.regulatory_keywords)
        if not regulatory_present:
            return None

        # Stress test results
        if "stress test" in text_lower:
            # Check for failure first (to catch "inadequate" before "adequate")
            if any(kw in text_lower for kw in ["fail", "inadequate", "deficient"]):
                return "stress_test_fail"
            elif any(kw in text_lower for kw in ["pass", "successful", "adequate"]):
                return "stress_test_pass"
            else:
                return "stress_test_pending"

        # Capital requirements
        if any(kw in text_lower for kw in ["capital requirement", "Basel", "tier 1"]):
            if any(kw in text_lower for kw in ["increase", "higher", "stricter"]):
                return "capital_requirement_increase"
            elif any(kw in text_lower for kw in ["decrease", "lower", "relaxed"]):
                return "capital_requirement_decrease"
            else:
                return "capital_requirement_change"

        # Compliance issues
        if any(kw in text_lower for kw in ["compliance", "violation", "fine", "penalty"]):
            return "compliance_issue"

        return "regulatory_change"
