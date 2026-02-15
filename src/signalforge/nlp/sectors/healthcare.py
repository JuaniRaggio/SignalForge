"""Healthcare sector analyzer."""

import re
from datetime import datetime

from signalforge.nlp.sectors.base import BaseSectorAnalyzer
from signalforge.nlp.sectors.schemas import HealthcareSignal, SignalStrength


class HealthcareAnalyzer(BaseSectorAnalyzer):
    """Analyzer for healthcare/pharma sector."""

    sector_name = "healthcare"

    def __init__(self) -> None:
        """Initialize healthcare analyzer with sector-specific keywords."""
        self.drug_phases = {
            "preclinical": ["preclinical", "animal study", "in vitro", "in vivo"],
            "phase1": ["phase 1", "phase I", "first-in-human", "safety study"],
            "phase2": ["phase 2", "phase II", "efficacy", "dose-ranging"],
            "phase3": ["phase 3", "phase III", "pivotal trial", "confirmatory"],
            "approved": ["FDA approved", "EMA approved", "market authorization", "NDA approved"],
        }
        self.fda_keywords = [
            "FDA", "approval", "rejection", "CRL", "PDUFA", "NDA", "BLA",
            "complete response letter", "priority review", "breakthrough designation"
        ]
        self.patent_keywords = [
            "patent", "exclusivity", "generic", "biosimilar", "patent expiration",
            "patent cliff", "IP protection", "orphan drug"
        ]
        self.clinical_keywords = [
            "trial", "endpoint", "efficacy", "safety", "adverse event",
            "statistically significant", "p-value", "enrollment"
        ]
        self.keywords = (
            [kw for phase_kws in self.drug_phases.values() for kw in phase_kws] +
            self.fda_keywords + self.patent_keywords + self.clinical_keywords
        )

    def get_sector_keywords(self) -> list[str]:
        """Return healthcare sector keywords.

        Returns:
            List of healthcare-specific keywords
        """
        return self.keywords

    def analyze(self, text: str, symbols: list[str]) -> list[HealthcareSignal]:  # type: ignore[override]
        """Extract healthcare signals.

        Analyzes text for:
        1. Drug pipeline stage detection
        2. FDA action tracking
        3. Patent status analysis
        4. Clinical trial results

        Args:
            text: Input text to analyze
            symbols: List of stock symbols mentioned

        Returns:
            List of healthcare signals
        """
        signals: list[HealthcareSignal] = []

        # Detect drug phase
        drug_phase_result = self.detect_drug_phase(text)
        if drug_phase_result:
            phase, confidence = drug_phase_result
            sentiment = self.calculate_sentiment(text, self.clinical_keywords)
            strength = self.map_to_signal_strength(sentiment)

            signals.append(HealthcareSignal(
                sector=self.sector_name,
                signal_type="drug_pipeline",
                strength=strength,
                confidence=confidence,
                description=f"Drug development at {phase} stage",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                drug_phase=phase,
            ))

        # Detect FDA action
        fda_result = self.detect_fda_action(text)
        if fda_result:
            action, action_strength = fda_result
            confidence = 0.85 if "approved" in action.lower() or "rejected" in action.lower() else 0.65

            signals.append(HealthcareSignal(
                sector=self.sector_name,
                signal_type="fda_action",
                strength=action_strength,
                confidence=confidence,
                description=f"FDA action: {action}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                fda_action=action,
            ))

        # Detect patent status
        patent_status = self.analyze_patent_status(text)
        if patent_status:
            sentiment = self.calculate_sentiment(text, self.patent_keywords)
            strength = self.map_to_signal_strength(sentiment)
            confidence = 0.7

            signals.append(HealthcareSignal(
                sector=self.sector_name,
                signal_type="patent_status",
                strength=strength,
                confidence=confidence,
                description=f"Patent status: {patent_status}",
                affected_symbols=symbols or self._extract_symbols_from_text(text),
                source_text=text[:500],
                timestamp=datetime.utcnow(),
                patent_status=patent_status,
            ))

        # Detect clinical trial results
        if any(kw.lower() in text.lower() for kw in self.clinical_keywords):
            sentiment = self.calculate_sentiment(text, self.clinical_keywords)
            if abs(sentiment) > 0.2:  # Only create signal if there's meaningful sentiment
                strength = self.map_to_signal_strength(sentiment)
                confidence = 0.6

                signals.append(HealthcareSignal(
                    sector=self.sector_name,
                    signal_type="clinical_trial",
                    strength=strength,
                    confidence=confidence,
                    description="Clinical trial results announced",
                    affected_symbols=symbols or self._extract_symbols_from_text(text),
                    source_text=text[:500],
                    timestamp=datetime.utcnow(),
                ))

        return signals

    def detect_drug_phase(self, text: str) -> tuple[str, float] | None:
        """Detect drug development phase.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (phase, confidence) or None
        """
        text_lower = text.lower()

        phase_scores: dict[str, int] = {}
        for phase, keywords in self.drug_phases.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                phase_scores[phase] = score

        if not phase_scores:
            return None

        max_phase = max(phase_scores, key=phase_scores.get)  # type: ignore
        max_score = phase_scores[max_phase]

        # Calculate confidence
        confidence = min(1.0, 0.5 + (max_score * 0.2))

        return max_phase, confidence

    def detect_fda_action(self, text: str) -> tuple[str, SignalStrength] | None:
        """Detect FDA actions (approval, rejection, CRL).

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (action, signal_strength) or None
        """
        text_lower = text.lower()

        # Approval patterns (very bullish)
        approval_patterns = [
            r'\bfda\s+approved?\b',
            r'\bfda\s+approval\b',
            r'\breceives?\s+fda\s+approval\b',
            r'\bgranted\s+approval\b',
            r'\bapproval\s+(?:granted|received)\b',
            r'\bmarket\s+authorization\b',
        ]
        for pattern in approval_patterns:
            if re.search(pattern, text_lower):
                return "approval", SignalStrength.STRONG_BULLISH

        # Rejection patterns (very bearish)
        rejection_patterns = [
            r'\bfda\s+rejected?\b',
            r'\brejection\s+letter\b',
            r'\bcomplete\s+response\s+letter\b',
            r'\bcrl\b',
        ]
        for pattern in rejection_patterns:
            if re.search(pattern, text_lower):
                return "rejection", SignalStrength.STRONG_BEARISH

        # Priority review (bullish)
        if re.search(r'\bpriority\s+review\b', text_lower):
            return "priority_review", SignalStrength.BULLISH

        # Breakthrough designation (bullish)
        if re.search(r'\bbreakthrough\s+designation\b', text_lower):
            return "breakthrough_designation", SignalStrength.BULLISH

        # PDUFA date mentioned (neutral but informative)
        if re.search(r'\bpdufa\s+(?:date|deadline)\b', text_lower):
            return "pdufa_date", SignalStrength.NEUTRAL

        return None

    def analyze_patent_status(self, text: str) -> str | None:
        """Analyze patent status signals.

        Args:
            text: Input text to analyze

        Returns:
            Patent status indicator or None
        """
        text_lower = text.lower()

        # Patent expiration (bearish for branded drugs)
        if any(kw in text_lower for kw in ["patent expiration", "patent cliff", "loss of exclusivity"]):
            return "expiring"

        # Patent granted (bullish)
        if any(kw in text_lower for kw in ["patent granted", "patent issued", "patent awarded"]):
            return "granted"

        # Generic competition (bearish for branded)
        if any(kw in text_lower for kw in ["generic", "biosimilar"]):
            return "generic_competition"

        # Patent protection maintained
        if any(kw in text_lower for kw in ["patent protection", "exclusivity", "ip protection"]):
            return "protected"

        # Orphan drug status (can be bullish due to exclusivity)
        if "orphan drug" in text_lower:
            return "orphan_status"

        return None
