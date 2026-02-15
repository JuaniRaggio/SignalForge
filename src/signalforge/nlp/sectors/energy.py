"""Energy sector analyzer."""

from datetime import datetime

from signalforge.nlp.sectors.base import BaseSectorAnalyzer
from signalforge.nlp.sectors.schemas import EnergySignal


class EnergyAnalyzer(BaseSectorAnalyzer):
    """Analyzer for energy sector."""

    sector_name = "energy"

    def __init__(self) -> None:
        """Initialize energy analyzer with sector-specific keywords."""
        self.commodities = [
            "oil", "natural gas", "crude", "WTI", "Brent", "LNG",
            "petroleum", "diesel", "gasoline", "fuel"
        ]
        self.renewables = [
            "solar", "wind", "hydro", "renewable", "clean energy",
            "photovoltaic", "turbine", "battery", "energy storage", "green energy"
        ]
        self.esg_keywords = [
            "ESG", "carbon", "emissions", "sustainability", "net zero",
            "decarbonization", "climate", "green", "environmental"
        ]
        self.price_keywords = [
            "price", "barrel", "per barrel", "commodity price", "oil price",
            "gas price", "trading at", "futures"
        ]
        self.keywords = (
            self.commodities + self.renewables + self.esg_keywords + self.price_keywords
        )

    def get_sector_keywords(self) -> list[str]:
        """Return energy sector keywords.

        Returns:
            List of energy-specific keywords
        """
        return self.keywords

    def analyze(self, text: str, symbols: list[str]) -> list[EnergySignal]:  # type: ignore[override]
        """Extract energy signals.

        Analyzes text for:
        1. Commodity price sensitivity
        2. Renewable transition signals
        3. ESG score impact
        4. Regulatory changes

        Args:
            text: Input text to analyze
            symbols: List of stock symbols mentioned

        Returns:
            List of energy signals
        """
        signals: list[EnergySignal] = []

        # Detect commodity type and price sensitivity
        commodity_result = self.detect_commodity_signals(text)
        if commodity_result:
            commodity, price_sensitivity = commodity_result
            sentiment = self.calculate_sentiment(text, self.commodities + self.price_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(EnergySignal(
                sector=self.sector_name,
                signal_type="commodity_price",
                strength=strength,
                confidence=0.75,
                description=f"Commodity price signal for {commodity}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                commodity=commodity,
                price_sensitivity=price_sensitivity,
            ))

        # Detect renewable energy transition signals
        if any(kw.lower() in text.lower() for kw in self.renewables):
            sentiment = self.calculate_sentiment(text, self.renewables)
            strength = self.map_to_signal_strength(sentiment)
            confidence = 0.7

            signals.append(EnergySignal(
                sector=self.sector_name,
                signal_type="renewable_transition",
                strength=strength,
                confidence=confidence,
                description="Renewable energy transition signal",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
            ))

        # Detect ESG impact signals
        esg_impact = self.analyze_esg_impact(text)
        if esg_impact is not None:
            strength = self.map_to_signal_strength(esg_impact)
            confidence = 0.65

            signals.append(EnergySignal(
                sector=self.sector_name,
                signal_type="esg_impact",
                strength=strength,
                confidence=confidence,
                description=f"ESG impact signal (score: {esg_impact:.2f})",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                esg_score_impact=esg_impact,
            ))

        # Detect regulatory signals
        if self.detect_regulatory_signal(text):
            sentiment = self.calculate_sentiment(text, ["regulation", "policy", "mandate"])
            strength = self.map_to_signal_strength(sentiment)
            confidence = 0.7

            signals.append(EnergySignal(
                sector=self.sector_name,
                signal_type="regulatory",
                strength=strength,
                confidence=confidence,
                description="Energy regulatory change detected",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
            ))

        return signals

    def detect_commodity_signals(self, text: str) -> tuple[str, float] | None:
        """Detect commodity type and price sensitivity.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (commodity, price_sensitivity) or None
        """
        text_lower = text.lower()

        # Identify which commodity is mentioned
        detected_commodity = None
        for commodity in self.commodities:
            if commodity.lower() in text_lower:
                detected_commodity = commodity
                break

        if not detected_commodity:
            return None

        # Calculate price sensitivity based on keywords
        high_sensitivity_keywords = [
            "surge", "spike", "soar", "plunge", "crash", "volatile",
            "significant increase", "sharp decline", "dramatic"
        ]
        medium_sensitivity_keywords = [
            "rise", "fall", "increase", "decrease", "change", "move"
        ]

        high_count = sum(1 for kw in high_sensitivity_keywords if kw in text_lower)
        medium_count = sum(1 for kw in medium_sensitivity_keywords if kw in text_lower)

        if high_count > 0:
            price_sensitivity = min(1.0, 0.7 + (high_count * 0.1))
        elif medium_count > 0:
            price_sensitivity = 0.5
        else:
            price_sensitivity = 0.3

        return detected_commodity, price_sensitivity

    def analyze_esg_impact(self, text: str) -> float | None:
        """Analyze ESG score impact from text.

        Args:
            text: Input text to analyze

        Returns:
            ESG impact score between -1.0 and 1.0, or None
        """
        text_lower = text.lower()

        # Check if ESG-related content is present
        esg_present = any(kw.lower() in text_lower for kw in self.esg_keywords)
        if not esg_present:
            return None

        # Positive ESG keywords
        positive_esg = [
            "reduce emissions", "carbon neutral", "net zero", "sustainable",
            "renewable", "clean energy", "green", "environmental leadership",
            "ESG improvement", "decarbonization"
        ]

        # Negative ESG keywords
        negative_esg = [
            "increase emissions", "environmental violation", "pollution",
            "carbon intensive", "fossil fuel expansion", "environmental damage",
            "ESG concerns", "greenwashing"
        ]

        positive_count = sum(1 for kw in positive_esg if kw in text_lower)
        negative_count = sum(1 for kw in negative_esg if kw in text_lower)

        if positive_count == 0 and negative_count == 0:
            return 0.0

        # Calculate net ESG impact
        net_impact = (positive_count - negative_count) / max(1, positive_count + negative_count)

        # Scale to -1.0 to 1.0 range
        return max(-1.0, min(1.0, net_impact))

    def detect_regulatory_signal(self, text: str) -> bool:
        """Detect regulatory signals in energy sector.

        Args:
            text: Input text to analyze

        Returns:
            True if regulatory signal detected
        """
        text_lower = text.lower()

        regulatory_keywords = [
            "regulation", "policy", "mandate", "law", "legislation",
            "government", "EPA", "department of energy", "DOE",
            "carbon tax", "emissions standard", "renewable mandate"
        ]

        return any(kw in text_lower for kw in regulatory_keywords)
