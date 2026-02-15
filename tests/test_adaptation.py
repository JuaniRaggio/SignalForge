"""Comprehensive tests for output adaptation layer."""

from uuid import uuid4

import pytest

from signalforge.adaptation import (
    AnalysisFormatter,
    GuidanceFormatter,
    InterpretationFormatter,
    OutputConfig,
    ProfileResolver,
    RawFormatter,
    TemplateEngine,
    create_adaptation_service,
    create_default_glossary,
)
from signalforge.adaptation.glossary import Glossary
from signalforge.adaptation.service import AdaptationService
from signalforge.models.user import ExperienceLevel, InvestmentHorizon, RiskTolerance, User


@pytest.fixture
def casual_user() -> User:
    """Create casual user for testing."""
    user = User(
        id=uuid4(),
        email="casual@test.com",
        hashed_password="hashed",
        username="casual_user",
        experience_level=ExperienceLevel.CASUAL,
        risk_tolerance=RiskTolerance.LOW,
        investment_horizon=InvestmentHorizon.LONG,
    )
    return user


@pytest.fixture
def informed_user() -> User:
    """Create informed user for testing."""
    user = User(
        id=uuid4(),
        email="informed@test.com",
        hashed_password="hashed",
        username="informed_user",
        experience_level=ExperienceLevel.INFORMED,
        risk_tolerance=RiskTolerance.MEDIUM,
        investment_horizon=InvestmentHorizon.MEDIUM,
    )
    return user


@pytest.fixture
def active_user() -> User:
    """Create active trader user for testing."""
    user = User(
        id=uuid4(),
        email="active@test.com",
        hashed_password="hashed",
        username="active_user",
        experience_level=ExperienceLevel.ACTIVE,
        risk_tolerance=RiskTolerance.HIGH,
        investment_horizon=InvestmentHorizon.SHORT,
    )
    return user


@pytest.fixture
def quant_user() -> User:
    """Create quant user for testing."""
    user = User(
        id=uuid4(),
        email="quant@test.com",
        hashed_password="hashed",
        username="quant_user",
        experience_level=ExperienceLevel.QUANT,
        risk_tolerance=RiskTolerance.HIGH,
        investment_horizon=InvestmentHorizon.SHORT,
    )
    return user


@pytest.fixture
def sample_signal_data() -> dict:
    """Sample signal data for testing."""
    return {
        "symbol": "AAPL",
        "type": "momentum",
        "signal": {
            "type": "buy",
            "direction": "bullish",
            "strength": "strong",
            "confidence": 0.85,
            "price_target": 180.50,
            "stop_loss": 165.25,
            "risk_reward": 3.2,
            "rationale": "Strong momentum with RSI oversold",
        },
        "metrics": {
            "price": 172.45,
            "change_percent": 0.025,
            "rsi": 68.5,
            "macd": 1.25,
            "volume": 45_000_000,
            "volatility": 0.28,
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.15,
            "alpha": 0.05,
            "beta": 1.12,
        },
        "indicators": {
            "moving_averages": {"sma_50": 170.20, "sma_200": 165.80},
            "oscillators": {"rsi": 68.5, "stochastic": 72.3},
            "trend": {"direction": "up", "strength": "strong"},
        },
        "recommendations": [
            {
                "action": "buy",
                "rationale": "Strong upward momentum",
                "priority": "high",
                "timeframe": "short",
                "risk_level": "medium",
            }
        ],
    }


class TestProfileResolver:
    """Test ProfileResolver functionality."""

    def test_resolve_casual_user(self, casual_user: User) -> None:
        """Test resolving casual user profile."""
        resolver = ProfileResolver()
        profile = resolver.resolve(casual_user)

        assert profile.user_id == casual_user.id
        assert profile.output_config.level == ExperienceLevel.CASUAL
        assert profile.output_config.include_glossary is True
        assert profile.output_config.include_raw_data is False
        assert profile.output_config.max_complexity == 3

    def test_resolve_informed_user(self, informed_user: User) -> None:
        """Test resolving informed user profile."""
        resolver = ProfileResolver()
        profile = resolver.resolve(informed_user)

        assert profile.output_config.level == ExperienceLevel.INFORMED
        assert profile.output_config.include_glossary is True
        assert profile.output_config.max_complexity == 6

    def test_resolve_active_user(self, active_user: User) -> None:
        """Test resolving active trader profile."""
        resolver = ProfileResolver()
        profile = resolver.resolve(active_user)

        assert profile.output_config.level == ExperienceLevel.ACTIVE
        assert profile.output_config.include_glossary is False
        assert profile.output_config.include_raw_data is True
        assert profile.output_config.max_complexity == 8

    def test_resolve_quant_user(self, quant_user: User) -> None:
        """Test resolving quant user profile."""
        resolver = ProfileResolver()
        profile = resolver.resolve(quant_user)

        assert profile.output_config.level == ExperienceLevel.QUANT
        assert profile.output_config.include_raw_data is True
        assert profile.output_config.max_complexity == 10

    def test_get_default_config_all_levels(self) -> None:
        """Test default config for all experience levels."""
        resolver = ProfileResolver()

        casual_config = resolver.get_default_config(ExperienceLevel.CASUAL)
        assert casual_config.level == ExperienceLevel.CASUAL
        assert casual_config.include_glossary is True

        quant_config = resolver.get_default_config(ExperienceLevel.QUANT)
        assert quant_config.level == ExperienceLevel.QUANT
        assert quant_config.include_glossary is False

    def test_override_config(self) -> None:
        """Test config override functionality."""
        resolver = ProfileResolver()
        base_config = resolver.get_default_config(ExperienceLevel.CASUAL)

        overrides = {"include_glossary": False, "max_complexity": 5}
        new_config = resolver.override_config(base_config, overrides)

        assert new_config.include_glossary is False
        assert new_config.max_complexity == 5
        assert new_config.level == ExperienceLevel.CASUAL

    def test_resolve_with_preferences(self, casual_user: User) -> None:
        """Test resolving user with notification preferences."""
        casual_user.notification_preferences = {
            "include_raw_data": True,
            "max_complexity": 7,
        }

        resolver = ProfileResolver()
        profile = resolver.resolve(casual_user)

        assert profile.output_config.include_raw_data is True
        assert profile.output_config.max_complexity == 7


class TestGlossary:
    """Test Glossary functionality."""

    def test_create_default_glossary(self) -> None:
        """Test creating default glossary."""
        glossary = create_default_glossary()
        terms = glossary.get_all_terms()

        assert len(terms) >= 50
        assert "eps" in terms
        assert "rsi" in terms
        assert "macd" in terms

    def test_get_definition(self) -> None:
        """Test getting term definition."""
        glossary = create_default_glossary()

        eps_def = glossary.get_definition("EPS")
        assert eps_def is not None
        assert "earnings" in eps_def.lower()

        rsi_def = glossary.get_definition("rsi")
        assert rsi_def is not None
        assert "momentum" in rsi_def.lower()

    def test_get_definition_not_found(self) -> None:
        """Test getting definition for non-existent term."""
        glossary = create_default_glossary()
        definition = glossary.get_definition("NONEXISTENT")
        assert definition is None

    def test_find_terms_in_text(self) -> None:
        """Test finding terms in text."""
        glossary = create_default_glossary()
        text = "The RSI indicator shows momentum, and the P/E ratio is high."

        found_terms = glossary.find_terms_in_text(text)

        assert "rsi" in found_terms
        assert "p/e" in found_terms or "p/e ratio" in found_terms

    def test_inject_tooltips(self) -> None:
        """Test injecting tooltips into text."""
        glossary = create_default_glossary()
        text = "The RSI is at 70."

        result = glossary.inject_tooltips(text)

        assert "RSI" in result
        assert "(" in result
        assert ")" in result
        assert len(result) > len(text)

    def test_custom_glossary(self) -> None:
        """Test creating custom glossary."""
        custom_terms = {"test_term": "Test definition", "another": "Another def"}
        glossary = Glossary(custom_terms)

        assert glossary.get_definition("test_term") == "Test definition"
        assert glossary.get_definition("ANOTHER") == "Another def"


class TestTemplateEngine:
    """Test TemplateEngine functionality."""

    def test_simplify_text_low_complexity(self) -> None:
        """Test text simplification for low complexity."""
        glossary = create_default_glossary()
        engine = TemplateEngine(glossary)

        text = "We need to leverage our position to mitigate risk and optimize returns."
        simplified = engine.simplify_text(text, target_complexity=2)

        assert "leverage" not in simplified.lower() or "borrowed" in simplified.lower()

    def test_add_context_casual(self) -> None:
        """Test adding context for casual users."""
        glossary = create_default_glossary()
        engine = TemplateEngine(glossary)

        data = {"key": "value"}
        enhanced = engine.add_context(data, ExperienceLevel.CASUAL)

        assert "_context" in enhanced
        assert enhanced["_context"]["show_explanations"] is True
        assert enhanced["_context"]["include_examples"] is True

    def test_add_context_quant(self) -> None:
        """Test adding context for quant users."""
        glossary = create_default_glossary()
        engine = TemplateEngine(glossary)

        data = {"key": "value"}
        enhanced = engine.add_context(data, ExperienceLevel.QUANT)

        assert "_context" in enhanced
        assert enhanced["_context"]["include_all_data"] is True
        assert enhanced["_context"]["preserve_precision"] is True

    def test_render_signal_template(self) -> None:
        """Test rendering signal template."""
        glossary = create_default_glossary()
        engine = TemplateEngine(glossary)

        config = OutputConfig(
            level=ExperienceLevel.CASUAL,
            include_glossary=False,
            include_raw_data=False,
            max_complexity=3,
        )

        context = {"title": "Buy Signal", "description": "Strong buy opportunity"}
        result = engine.render("signal", context, config)

        assert "Signal: Buy Signal" in result or result == ""


class TestRawFormatter:
    """Test RawFormatter functionality."""

    def test_format_preserves_all_data(self, sample_signal_data: dict) -> None:
        """Test that raw formatter preserves all data."""
        formatter = RawFormatter()
        result = formatter.format(sample_signal_data)

        assert result["symbol"] == sample_signal_data["symbol"]
        assert result["metrics"] == sample_signal_data["metrics"]
        assert "_meta" in result
        assert result["_meta"]["formatter"] == "raw"


class TestAnalysisFormatter:
    """Test AnalysisFormatter functionality."""

    def test_format_includes_technical_metrics(self, sample_signal_data: dict) -> None:
        """Test that analysis formatter includes technical metrics."""
        formatter = AnalysisFormatter()
        result = formatter.format(sample_signal_data)

        assert "metrics" in result
        assert "signal" in result
        assert "_meta" in result
        assert result["_meta"]["experience_level"] == "active"

    def test_format_signal(self, sample_signal_data: dict) -> None:
        """Test signal formatting."""
        formatter = AnalysisFormatter()
        result = formatter.format(sample_signal_data)

        assert "signal" in result
        assert "confidence" in result["signal"]
        assert "%" in result["signal"]["confidence"]


class TestInterpretationFormatter:
    """Test InterpretationFormatter functionality."""

    def test_format_adds_context(self, sample_signal_data: dict) -> None:
        """Test that interpretation formatter adds context."""
        formatter = InterpretationFormatter()
        result = formatter.format(sample_signal_data)

        assert "key_insights" in result
        assert "market_context" in result
        assert "_meta" in result
        assert result["_meta"]["focus"] == "context_and_significance"

    def test_extract_key_insights(self, sample_signal_data: dict) -> None:
        """Test key insights extraction."""
        formatter = InterpretationFormatter()
        result = formatter.format(sample_signal_data)

        assert "key_insights" in result
        assert len(result["key_insights"]) > 0
        assert any("interpretation" in insight for insight in result["key_insights"])


class TestGuidanceFormatter:
    """Test GuidanceFormatter functionality."""

    def test_format_simplifies_content(self, sample_signal_data: dict) -> None:
        """Test that guidance formatter simplifies content."""
        formatter = GuidanceFormatter()
        result = formatter.format(sample_signal_data)

        assert "simple_summary" in result
        assert "what_this_means" in result
        assert "what_to_do" in result
        assert "_meta" in result
        assert result["_meta"]["focus"] == "simple_explanations"

    def test_simple_summary(self, sample_signal_data: dict) -> None:
        """Test simple summary creation."""
        formatter = GuidanceFormatter()
        result = formatter.format(sample_signal_data)

        assert "simple_summary" in result
        summary = result["simple_summary"]
        assert "outlook" in summary
        assert "simple_explanation" in summary
        assert "reliability" in summary

    def test_what_to_do(self, sample_signal_data: dict) -> None:
        """Test actionable recommendations."""
        formatter = GuidanceFormatter()
        result = formatter.format(sample_signal_data)

        assert "what_to_do" in result
        actions = result["what_to_do"]
        assert len(actions) > 0
        assert "suggestion" in actions[0]
        assert "reason" in actions[0]


class TestAdaptationService:
    """Test AdaptationService functionality."""

    def test_create_service(self) -> None:
        """Test creating adaptation service."""
        service = create_adaptation_service()
        assert isinstance(service, AdaptationService)

    def test_adapt_for_casual_user(
        self, casual_user: User, sample_signal_data: dict
    ) -> None:
        """Test adapting content for casual user."""
        service = create_adaptation_service()
        result = service.adapt(sample_signal_data, casual_user)

        assert "_output_config" in result
        assert result["_output_config"]["level"] == "casual"
        assert "simple_summary" in result

    def test_adapt_for_quant_user(
        self, quant_user: User, sample_signal_data: dict
    ) -> None:
        """Test adapting content for quant user."""
        service = create_adaptation_service()
        result = service.adapt(sample_signal_data, quant_user)

        assert "_output_config" in result
        assert result["_output_config"]["level"] == "quant"
        assert result["_meta"]["formatter"] == "raw"

    def test_adapt_signal(self, informed_user: User, sample_signal_data: dict) -> None:
        """Test adapting signal specifically."""
        service = create_adaptation_service()
        result = service.adapt_signal(sample_signal_data, informed_user)

        assert "_output_config" in result
        assert result["_output_config"]["level"] == "informed"

    def test_adapt_news(self, active_user: User) -> None:
        """Test adapting news article."""
        service = create_adaptation_service()
        news_data = {
            "title": "Market Update",
            "summary": "The RSI shows strong momentum in tech sector.",
            "timestamp": "2024-01-01T10:00:00",
        }

        result = service.adapt_news(news_data, active_user)

        assert "_output_config" in result
        assert result["_output_config"]["level"] == "active"

    def test_get_formatter_for_user(self, casual_user: User) -> None:
        """Test getting formatter for user."""
        service = create_adaptation_service()
        formatter = service.get_formatter_for_user(casual_user)

        assert isinstance(formatter, GuidanceFormatter)

    def test_adapt_includes_preferences(
        self, casual_user: User, sample_signal_data: dict
    ) -> None:
        """Test that adapted content includes user preferences."""
        casual_user.preferred_sectors = ["technology", "healthcare"]
        casual_user.watchlist = ["AAPL", "GOOGL"]

        service = create_adaptation_service()
        result = service.adapt(sample_signal_data, casual_user)

        assert "_user_preferences" in result
        assert "preferred_sectors" in result["_user_preferences"]
        assert "technology" in result["_user_preferences"]["preferred_sectors"]


class TestIntegration:
    """Integration tests for full adaptation flow."""

    def test_full_adaptation_flow_casual(
        self, casual_user: User, sample_signal_data: dict
    ) -> None:
        """Test full adaptation flow for casual user."""
        service = create_adaptation_service()

        result = service.adapt_signal(sample_signal_data, casual_user)

        assert result is not None
        assert "_output_config" in result
        assert result["_output_config"]["level"] == "casual"
        assert result["_output_config"]["include_glossary"] is True

    def test_full_adaptation_flow_quant(
        self, quant_user: User, sample_signal_data: dict
    ) -> None:
        """Test full adaptation flow for quant user."""
        service = create_adaptation_service()

        result = service.adapt_signal(sample_signal_data, quant_user)

        assert result is not None
        assert result["_meta"]["formatter"] == "raw"
        assert "metrics" in result
        assert result["metrics"] == sample_signal_data["metrics"]

    def test_different_users_same_data(self, sample_signal_data: dict) -> None:
        """Test that same data adapts differently for different users."""
        service = create_adaptation_service()

        casual_user = User(
            id=uuid4(),
            email="casual@test.com",
            hashed_password="hashed",
            username="casual",
            experience_level=ExperienceLevel.CASUAL,
            risk_tolerance=RiskTolerance.LOW,
            investment_horizon=InvestmentHorizon.LONG,
        )

        quant_user = User(
            id=uuid4(),
            email="quant@test.com",
            hashed_password="hashed",
            username="quant",
            experience_level=ExperienceLevel.QUANT,
            risk_tolerance=RiskTolerance.HIGH,
            investment_horizon=InvestmentHorizon.SHORT,
        )

        casual_result = service.adapt(sample_signal_data, casual_user)
        quant_result = service.adapt(sample_signal_data, quant_user)

        assert casual_result["_output_config"]["level"] == "casual"
        assert quant_result["_output_config"]["level"] == "quant"
        assert "simple_summary" in casual_result
        assert "simple_summary" not in quant_result
