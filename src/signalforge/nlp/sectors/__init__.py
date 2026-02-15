"""Sector intelligence modules for NLP pipeline."""

from signalforge.nlp.sectors.base import BaseSectorAnalyzer
from signalforge.nlp.sectors.consumer import ConsumerAnalyzer
from signalforge.nlp.sectors.energy import EnergyAnalyzer
from signalforge.nlp.sectors.financial import FinancialAnalyzer
from signalforge.nlp.sectors.healthcare import HealthcareAnalyzer
from signalforge.nlp.sectors.schemas import (
    ConsumerSignal,
    EnergySignal,
    FinancialSignal,
    HealthcareSignal,
    SectorSignal,
    SignalStrength,
    TechnologySignal,
)
from signalforge.nlp.sectors.technology import TechnologyAnalyzer

__all__ = [
    "BaseSectorAnalyzer",
    "TechnologyAnalyzer",
    "HealthcareAnalyzer",
    "EnergyAnalyzer",
    "FinancialAnalyzer",
    "ConsumerAnalyzer",
    "SectorAnalyzerFactory",
    "SectorSignal",
    "TechnologySignal",
    "HealthcareSignal",
    "EnergySignal",
    "FinancialSignal",
    "ConsumerSignal",
    "SignalStrength",
]


class SectorAnalyzerFactory:
    """Factory for creating sector-specific analyzers."""

    _analyzers: dict[str, type[BaseSectorAnalyzer]] = {
        "technology": TechnologyAnalyzer,
        "healthcare": HealthcareAnalyzer,
        "energy": EnergyAnalyzer,
        "financial": FinancialAnalyzer,
        "consumer": ConsumerAnalyzer,
    }

    @classmethod
    def get_analyzer(cls, sector: str) -> BaseSectorAnalyzer:
        """Get analyzer for specified sector.

        Args:
            sector: Sector name (technology, healthcare, energy, financial, consumer)

        Returns:
            Sector analyzer instance

        Raises:
            ValueError: If sector is not supported
        """
        sector_lower = sector.lower()
        if sector_lower not in cls._analyzers:
            available = ", ".join(cls._analyzers.keys())
            msg = f"Unsupported sector '{sector}'. Available sectors: {available}"
            raise ValueError(msg)

        analyzer_class = cls._analyzers[sector_lower]
        return analyzer_class()

    @classmethod
    def get_all_analyzers(cls) -> dict[str, BaseSectorAnalyzer]:
        """Get all available sector analyzers.

        Returns:
            Dictionary mapping sector names to analyzer instances
        """
        return {sector: analyzer_class() for sector, analyzer_class in cls._analyzers.items()}

    @classmethod
    def get_supported_sectors(cls) -> list[str]:
        """Get list of supported sectors.

        Returns:
            List of sector names
        """
        return list(cls._analyzers.keys())

    @classmethod
    def register_analyzer(cls, sector: str, analyzer_class: type[BaseSectorAnalyzer]) -> None:
        """Register a custom sector analyzer.

        Args:
            sector: Sector name
            analyzer_class: Analyzer class to register
        """
        cls._analyzers[sector.lower()] = analyzer_class
