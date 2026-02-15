"""Output Adaptation Layer for user-specific content rendering."""

from signalforge.adaptation.formatters import (
    AnalysisFormatter,
    BaseFormatter,
    GuidanceFormatter,
    InterpretationFormatter,
    RawFormatter,
)
from signalforge.adaptation.glossary import Glossary, create_default_glossary
from signalforge.adaptation.profile_resolver import (
    OutputConfig,
    ProfileResolver,
    ResolvedProfile,
)
from signalforge.adaptation.service import AdaptationService
from signalforge.adaptation.template_engine import TemplateEngine
from signalforge.models.user import ExperienceLevel

__all__ = [
    "OutputConfig",
    "ResolvedProfile",
    "ProfileResolver",
    "Glossary",
    "create_default_glossary",
    "TemplateEngine",
    "BaseFormatter",
    "RawFormatter",
    "AnalysisFormatter",
    "InterpretationFormatter",
    "GuidanceFormatter",
    "AdaptationService",
]


def create_adaptation_service() -> AdaptationService:
    """
    Create fully configured adaptation service.

    Returns:
        AdaptationService instance with all components
    """
    glossary = create_default_glossary()
    template_engine = TemplateEngine(glossary)
    profile_resolver = ProfileResolver()

    formatters = {
        ExperienceLevel.CASUAL: GuidanceFormatter(),
        ExperienceLevel.INFORMED: InterpretationFormatter(),
        ExperienceLevel.ACTIVE: AnalysisFormatter(),
        ExperienceLevel.QUANT: RawFormatter(),
    }

    return AdaptationService(profile_resolver, template_engine, formatters)
