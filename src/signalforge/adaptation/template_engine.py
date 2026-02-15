"""Template engine for adaptive content rendering."""

import re
from typing import Any

import structlog

from signalforge.adaptation.glossary import Glossary
from signalforge.adaptation.profile_resolver import OutputConfig
from signalforge.models.user import ExperienceLevel

logger = structlog.get_logger(__name__)


class TemplateEngine:
    """
    Template engine for rendering adaptive content based on user profile.

    Provides text simplification, context addition, and glossary integration.
    """

    def __init__(self, glossary: Glossary) -> None:
        """
        Initialize template engine.

        Args:
            glossary: Glossary instance for term definitions
        """
        self._glossary = glossary
        self._logger = logger.bind(component="template_engine")
        self._logger.info("template_engine_initialized")

    def render(
        self, template_name: str, context: dict[str, Any], config: OutputConfig
    ) -> str:
        """
        Render template with context adapted to user level.

        Args:
            template_name: Name of template to render
            context: Template context data
            config: Output configuration

        Returns:
            Rendered string adapted to user level
        """
        self._logger.debug(
            "rendering_template",
            template=template_name,
            level=config.level.value,
        )

        enhanced_context = self.add_context(context, config.level)

        text = self._render_template(template_name, enhanced_context)

        if config.max_complexity < 10:
            text = self.simplify_text(text, config.max_complexity)

        if config.include_glossary:
            text = self._glossary.inject_tooltips(text)

        return text

    def simplify_text(self, text: str, target_complexity: int) -> str:
        """
        Simplify text based on target complexity level.

        Args:
            text: Original text
            target_complexity: Target complexity (1-10, lower is simpler)

        Returns:
            Simplified text
        """
        self._logger.debug(
            "simplifying_text",
            original_length=len(text),
            target_complexity=target_complexity,
        )

        result = text

        if target_complexity <= 3:
            result = self._simplify_sentences(result)
            result = self._replace_jargon(result)

        if target_complexity <= 6:
            result = self._shorten_paragraphs(result)

        return result

    def add_context(
        self, data: dict[str, Any], level: ExperienceLevel
    ) -> dict[str, Any]:
        """
        Add contextual information based on experience level.

        Args:
            data: Original data dictionary
            level: User experience level

        Returns:
            Enhanced data with context added
        """
        self._logger.debug("adding_context", level=level.value)

        enhanced = data.copy()

        if level == ExperienceLevel.CASUAL:
            enhanced["_context"] = {
                "show_explanations": True,
                "include_examples": True,
                "simplify_language": True,
            }
        elif level == ExperienceLevel.INFORMED:
            enhanced["_context"] = {
                "show_market_context": True,
                "include_comparisons": True,
                "explain_significance": True,
            }
        elif level == ExperienceLevel.ACTIVE:
            enhanced["_context"] = {
                "show_technical_details": True,
                "include_metrics": True,
                "highlight_actionable": True,
            }
        elif level == ExperienceLevel.QUANT:
            enhanced["_context"] = {
                "include_all_data": True,
                "show_calculations": True,
                "preserve_precision": True,
            }

        return enhanced

    def _render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Internal template rendering method.

        Args:
            template_name: Template name
            context: Rendering context

        Returns:
            Rendered string
        """
        templates = {
            "signal": self._render_signal_template,
            "news": self._render_news_template,
            "metric": self._render_metric_template,
        }

        renderer = templates.get(template_name, self._render_default_template)
        return renderer(context)

    def _render_signal_template(self, context: dict[str, Any]) -> str:
        """Render signal template."""
        parts = []

        if "title" in context:
            parts.append(f"Signal: {context['title']}")

        if "description" in context:
            parts.append(context["description"])

        if "metrics" in context and context.get("_context", {}).get("include_metrics"):
            parts.append(f"Metrics: {context['metrics']}")

        return "\n".join(parts)

    def _render_news_template(self, context: dict[str, Any]) -> str:
        """Render news template."""
        parts = []

        if "headline" in context:
            parts.append(f"News: {context['headline']}")

        if "summary" in context:
            parts.append(context["summary"])

        if "impact" in context and context.get("_context", {}).get("explain_significance"):
            parts.append(f"Impact: {context['impact']}")

        return "\n".join(parts)

    def _render_metric_template(self, context: dict[str, Any]) -> str:
        """Render metric template."""
        parts = []

        if "name" in context:
            parts.append(f"Metric: {context['name']}")

        if "value" in context:
            parts.append(f"Value: {context['value']}")

        if "explanation" in context and context.get("_context", {}).get(
            "show_explanations"
        ):
            parts.append(f"Explanation: {context['explanation']}")

        return "\n".join(parts)

    def _render_default_template(self, context: dict[str, Any]) -> str:
        """Default template renderer."""
        return str(context)

    def _simplify_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones."""
        sentences = re.split(r'([.!?]+)', text)
        result = []

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""

            if len(sentence.split()) > 15:
                words = sentence.split()
                mid = len(words) // 2

                for j in range(mid, min(mid + 5, len(words))):
                    if words[j] in {"and", "but", "or", "because", "since", "while"}:
                        mid = j
                        break

                part1 = " ".join(words[:mid])
                part2 = " ".join(words[mid:])
                result.append(part1 + "." + " " + part2.capitalize() + punctuation)
            else:
                result.append(sentence + punctuation)

        return "".join(result)

    def _replace_jargon(self, text: str) -> str:
        """Replace financial jargon with simpler terms."""
        replacements = {
            r'\bleverage\b': "borrowed money",
            r'\bvolatile\b': "unstable",
            r'\bliquidate\b': "sell",
            r'\bacquisition\b': "purchase",
            r'\bmitigate\b': "reduce",
            r'\boptimize\b': "improve",
            r'\butilize\b': "use",
            r'\bfacilitate\b': "help",
            r'\bimplement\b': "carry out",
        }

        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def _shorten_paragraphs(self, text: str) -> str:
        """Break long paragraphs into shorter ones."""
        paragraphs = text.split("\n\n")
        result = []

        for para in paragraphs:
            sentences = re.split(r'([.!?]+\s+)', para)
            if len(sentences) > 8:
                mid = len(sentences) // 2
                part1 = "".join(sentences[:mid])
                part2 = "".join(sentences[mid:])
                result.append(part1.strip())
                result.append(part2.strip())
            else:
                result.append(para)

        return "\n\n".join(result)
