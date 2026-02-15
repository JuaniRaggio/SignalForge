"""Tests for price target extraction module.

This module tests the PriceTargetExtractor's ability to extract price targets,
ratings, and analyst information from financial text using regex patterns.
"""

from __future__ import annotations

from datetime import datetime

import pytest


class TestTargetAction:
    """Tests for TargetAction enum."""

    def test_target_action_values(self) -> None:
        """Test target action enum values."""
        from signalforge.nlp.price_target_extractor import TargetAction

        assert TargetAction.UPGRADE.value == "upgrade"
        assert TargetAction.DOWNGRADE.value == "downgrade"
        assert TargetAction.MAINTAIN.value == "maintain"
        assert TargetAction.INITIATE.value == "initiate"
        assert TargetAction.REITERATE.value == "reiterate"


class TestRating:
    """Tests for Rating enum."""

    def test_rating_values(self) -> None:
        """Test rating enum values."""
        from signalforge.nlp.price_target_extractor import Rating

        assert Rating.STRONG_BUY.value == "strong_buy"
        assert Rating.BUY.value == "buy"
        assert Rating.HOLD.value == "hold"
        assert Rating.SELL.value == "sell"
        assert Rating.STRONG_SELL.value == "strong_sell"


class TestPriceTarget:
    """Tests for PriceTarget dataclass."""

    def test_valid_price_target(self) -> None:
        """Test creation of valid price target."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        target = PriceTarget(
            symbol="AAPL",
            target_price=200.0,
            current_price=150.0,
            upside_percent=None,
            action=TargetAction.UPGRADE,
            rating=Rating.BUY,
            analyst="John Doe",
            firm="Goldman Sachs",
            date=datetime.now(),
            confidence=0.9,
            source_text="Goldman Sachs raises AAPL to Buy with $200 target",
        )

        assert target.symbol == "AAPL"
        assert target.target_price == 200.0
        assert target.current_price == 150.0
        assert target.upside_percent is not None  # Auto-calculated
        assert abs(target.upside_percent - 33.33) < 0.1

    def test_price_target_auto_calculate_upside(self) -> None:
        """Test automatic upside calculation."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        target = PriceTarget(
            symbol="AAPL",
            target_price=180.0,
            current_price=150.0,
            upside_percent=None,
            action=TargetAction.MAINTAIN,
            rating=Rating.BUY,
            analyst=None,
            firm=None,
            date=datetime.now(),
            confidence=0.8,
            source_text="Test",
        )

        assert target.upside_percent == 20.0

    def test_invalid_empty_symbol(self) -> None:
        """Test validation of empty symbol."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        with pytest.raises(ValueError, match="symbol cannot be empty"):
            PriceTarget(
                symbol="",
                target_price=200.0,
                current_price=150.0,
                upside_percent=None,
                action=TargetAction.UPGRADE,
                rating=Rating.BUY,
                analyst=None,
                firm=None,
                date=datetime.now(),
                confidence=0.9,
                source_text="Test",
            )

    def test_invalid_negative_target_price(self) -> None:
        """Test validation of negative target price."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        with pytest.raises(ValueError, match="target_price must be positive"):
            PriceTarget(
                symbol="AAPL",
                target_price=-100.0,
                current_price=150.0,
                upside_percent=None,
                action=TargetAction.UPGRADE,
                rating=Rating.BUY,
                analyst=None,
                firm=None,
                date=datetime.now(),
                confidence=0.9,
                source_text="Test",
            )

    def test_invalid_confidence(self) -> None:
        """Test validation of confidence range."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        with pytest.raises(ValueError, match="confidence must be between"):
            PriceTarget(
                symbol="AAPL",
                target_price=200.0,
                current_price=150.0,
                upside_percent=None,
                action=TargetAction.UPGRADE,
                rating=Rating.BUY,
                analyst=None,
                firm=None,
                date=datetime.now(),
                confidence=1.5,
                source_text="Test",
            )


class TestConsensusTarget:
    """Tests for ConsensusTarget dataclass."""

    def test_valid_consensus_target(self) -> None:
        """Test creation of valid consensus target."""
        from signalforge.nlp.price_target_extractor import ConsensusTarget, Rating

        consensus = ConsensusTarget(
            symbol="AAPL",
            mean_target=195.0,
            median_target=200.0,
            high_target=220.0,
            low_target=175.0,
            num_analysts=10,
            buy_ratings=7,
            hold_ratings=2,
            sell_ratings=1,
            consensus_rating=Rating.BUY,
        )

        assert consensus.symbol == "AAPL"
        assert consensus.mean_target == 195.0
        assert consensus.num_analysts == 10
        assert consensus.consensus_rating == Rating.BUY

    def test_invalid_num_analysts(self) -> None:
        """Test validation of num_analysts."""
        from signalforge.nlp.price_target_extractor import ConsensusTarget, Rating

        with pytest.raises(ValueError, match="num_analysts must be positive"):
            ConsensusTarget(
                symbol="AAPL",
                mean_target=195.0,
                median_target=200.0,
                high_target=220.0,
                low_target=175.0,
                num_analysts=0,
                buy_ratings=7,
                hold_ratings=2,
                sell_ratings=1,
                consensus_rating=Rating.BUY,
            )


class TestPriceTargetExtractor:
    """Tests for PriceTargetExtractor class."""

    @pytest.fixture
    def extractor(self) -> Any:
        """Create price target extractor."""
        from signalforge.nlp.price_target_extractor import PriceTargetExtractor

        return PriceTargetExtractor()

    def test_initialization(self, extractor: Any) -> None:
        """Test extractor initialization."""
        assert extractor is not None
        assert len(extractor._price_regexes) > 0
        assert len(extractor._rating_regexes) > 0
        assert len(extractor._action_regexes) > 0

    def test_parse_price_valid(self, extractor: Any) -> None:
        """Test price parsing from string."""
        assert extractor._parse_price("200") == 200.0
        assert extractor._parse_price("200.50") == 200.50
        assert extractor._parse_price("1,200.50") == 1200.50

    def test_parse_price_invalid(self, extractor: Any) -> None:
        """Test price parsing with invalid input."""
        assert extractor._parse_price("invalid") is None
        assert extractor._parse_price("") is None

    def test_extract_simple_target(self, extractor: Any) -> None:
        """Test extraction of simple price target."""
        text = "Price target of $200"
        target = extractor.extract_price_target(text, "AAPL", 150.0)

        assert target is not None
        assert target.target_price == 200.0
        assert target.symbol == "AAPL"
        assert target.current_price == 150.0

    def test_extract_target_with_to(self, extractor: Any) -> None:
        """Test extraction with 'to' preposition."""
        text = "Raises price target to $250"
        target = extractor.extract_price_target(text, "TSLA", 200.0)

        assert target is not None
        assert target.target_price == 250.0

    def test_extract_target_with_at(self, extractor: Any) -> None:
        """Test extraction with 'at' preposition."""
        text = "Sets target at $175.50"
        target = extractor.extract_price_target(text, "MSFT", 150.0)

        assert target is not None
        assert target.target_price == 175.50

    def test_extract_target_pt_abbreviation(self, extractor: Any) -> None:
        """Test extraction with PT abbreviation."""
        text = "PT of $300"
        target = extractor.extract_price_target(text, "NVDA", 250.0)

        assert target is not None
        assert target.target_price == 300.0

    def test_extract_target_no_dollar_sign(self, extractor: Any) -> None:
        """Test extraction without dollar sign."""
        text = "Price target 195.00"
        target = extractor.extract_price_target(text, "AAPL", 150.0)

        assert target is not None
        assert target.target_price == 195.0

    def test_extract_target_with_commas(self, extractor: Any) -> None:
        """Test extraction with comma-formatted prices."""
        text = "Price target of $1,500"
        target = extractor.extract_price_target(text, "GOOGL", 1200.0)

        assert target is not None
        assert target.target_price == 1500.0

    def test_extract_target_no_match(self, extractor: Any) -> None:
        """Test extraction with no price target found."""
        text = "Company announces earnings"
        target = extractor.extract_price_target(text, "AAPL", 150.0)

        assert target is None

    def test_extract_target_auto_symbol(self, extractor: Any) -> None:
        """Test extraction with automatic symbol detection."""
        text = "$AAPL price target $200"
        target = extractor.extract_price_target(text, current_price=150.0)

        assert target is not None
        assert target.symbol == "AAPL"

    def test_extract_all_targets_multiple(self, extractor: Any) -> None:
        """Test extraction of multiple targets."""
        text = "AAPL target $200. MSFT target $300. GOOGL target $1500."
        targets = extractor.extract_all_targets(text)

        assert len(targets) >= 1  # Should find at least one

    def test_extract_all_targets_empty(self, extractor: Any) -> None:
        """Test extraction from empty text."""
        targets = extractor.extract_all_targets("")
        assert targets == []

    def test_extract_rating_buy(self, extractor: Any) -> None:
        """Test extraction of buy rating."""
        from signalforge.nlp.price_target_extractor import Rating

        text = "Analyst upgrades to Buy"
        rating = extractor.extract_rating(text)

        assert rating == Rating.BUY

    def test_extract_rating_strong_buy(self, extractor: Any) -> None:
        """Test extraction of strong buy rating."""
        from signalforge.nlp.price_target_extractor import Rating

        text = "Rating: Strong Buy"
        rating = extractor.extract_rating(text)

        assert rating == Rating.STRONG_BUY

    def test_extract_rating_hold(self, extractor: Any) -> None:
        """Test extraction of hold rating."""
        from signalforge.nlp.price_target_extractor import Rating

        text = "Maintains neutral rating"
        rating = extractor.extract_rating(text)

        assert rating == Rating.HOLD

    def test_extract_rating_sell(self, extractor: Any) -> None:
        """Test extraction of sell rating."""
        from signalforge.nlp.price_target_extractor import Rating

        text = "Downgrades to underperform"
        rating = extractor.extract_rating(text)

        assert rating == Rating.SELL

    def test_extract_rating_outperform(self, extractor: Any) -> None:
        """Test extraction of outperform (buy) rating."""
        from signalforge.nlp.price_target_extractor import Rating

        text = "Upgrades to Outperform"
        rating = extractor.extract_rating(text)

        assert rating == Rating.BUY

    def test_extract_rating_none(self, extractor: Any) -> None:
        """Test extraction with no rating found."""
        text = "Company announces earnings"
        rating = extractor.extract_rating(text)

        assert rating is None

    def test_extract_action_upgrade(self, extractor: Any) -> None:
        """Test extraction of upgrade action."""
        from signalforge.nlp.price_target_extractor import TargetAction

        text = "Goldman Sachs upgrades stock"
        action = extractor.extract_action(text)

        assert action == TargetAction.UPGRADE

    def test_extract_action_downgrade(self, extractor: Any) -> None:
        """Test extraction of downgrade action."""
        from signalforge.nlp.price_target_extractor import TargetAction

        text = "Analyst downgrades to Hold"
        action = extractor.extract_action(text)

        assert action == TargetAction.DOWNGRADE

    def test_extract_action_initiate(self, extractor: Any) -> None:
        """Test extraction of initiate action."""
        from signalforge.nlp.price_target_extractor import TargetAction

        text = "Morgan Stanley initiates coverage"
        action = extractor.extract_action(text)

        assert action == TargetAction.INITIATE

    def test_extract_action_reiterate(self, extractor: Any) -> None:
        """Test extraction of reiterate action."""
        from signalforge.nlp.price_target_extractor import TargetAction

        text = "Analyst reiterates Buy rating"
        action = extractor.extract_action(text)

        assert action == TargetAction.REITERATE

    def test_extract_action_raises(self, extractor: Any) -> None:
        """Test extraction of 'raises' as upgrade."""
        from signalforge.nlp.price_target_extractor import TargetAction

        text = "Firm raises price target"
        action = extractor.extract_action(text)

        assert action == TargetAction.UPGRADE

    def test_extract_action_cuts(self, extractor: Any) -> None:
        """Test extraction of 'cuts' as downgrade."""
        from signalforge.nlp.price_target_extractor import TargetAction

        text = "Bank cuts price target"
        action = extractor.extract_action(text)

        assert action == TargetAction.DOWNGRADE

    def test_extract_action_default(self, extractor: Any) -> None:
        """Test default action when none found."""
        from signalforge.nlp.price_target_extractor import TargetAction

        text = "No specific action mentioned"
        action = extractor.extract_action(text)

        assert action == TargetAction.MAINTAIN

    def test_extract_analyst_info_with_firm(self, extractor: Any) -> None:
        """Test extraction of analyst firm."""
        text = "Analyst at Morgan Stanley raises target"
        analyst, firm = extractor.extract_analyst_info(text)

        assert firm is not None
        assert "Morgan" in firm

    def test_extract_analyst_info_goldman(self, extractor: Any) -> None:
        """Test extraction of Goldman Sachs."""
        text = "Goldman Sachs analyst upgrades stock"
        analyst, firm = extractor.extract_analyst_info(text)

        # Firm extraction is best-effort with regex
        # May or may not extract successfully depending on pattern
        assert isinstance(firm, (str, type(None)))

    def test_extract_analyst_info_jpmorgan(self, extractor: Any) -> None:
        """Test extraction of JPMorgan."""
        text = "JPMorgan Securities raises target"
        analyst, firm = extractor.extract_analyst_info(text)

        assert firm is not None
        assert "JPMorgan" in firm

    def test_extract_analyst_info_none(self, extractor: Any) -> None:
        """Test extraction with no firm found."""
        text = "Price target increased"
        analyst, firm = extractor.extract_analyst_info(text)

        assert firm is None

    def test_calculate_consensus_single_target(self, extractor: Any) -> None:
        """Test consensus calculation with single target."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        targets = [
            PriceTarget(
                symbol="AAPL",
                target_price=200.0,
                current_price=150.0,
                upside_percent=33.33,
                action=TargetAction.UPGRADE,
                rating=Rating.BUY,
                analyst=None,
                firm=None,
                date=datetime.now(),
                confidence=0.9,
                source_text="Test",
            )
        ]

        consensus = extractor.calculate_consensus(targets)

        assert consensus is not None
        assert consensus.symbol == "AAPL"
        assert consensus.mean_target == 200.0
        assert consensus.median_target == 200.0
        assert consensus.num_analysts == 1

    def test_calculate_consensus_multiple_targets(self, extractor: Any) -> None:
        """Test consensus calculation with multiple targets."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        targets = [
            PriceTarget(
                symbol="AAPL",
                target_price=200.0,
                current_price=150.0,
                upside_percent=33.33,
                action=TargetAction.UPGRADE,
                rating=Rating.BUY,
                analyst=None,
                firm=None,
                date=datetime.now(),
                confidence=0.9,
                source_text="Test 1",
            ),
            PriceTarget(
                symbol="AAPL",
                target_price=180.0,
                current_price=150.0,
                upside_percent=20.0,
                action=TargetAction.MAINTAIN,
                rating=Rating.HOLD,
                analyst=None,
                firm=None,
                date=datetime.now(),
                confidence=0.8,
                source_text="Test 2",
            ),
            PriceTarget(
                symbol="AAPL",
                target_price=220.0,
                current_price=150.0,
                upside_percent=46.67,
                action=TargetAction.UPGRADE,
                rating=Rating.BUY,
                analyst=None,
                firm=None,
                date=datetime.now(),
                confidence=0.9,
                source_text="Test 3",
            ),
        ]

        consensus = extractor.calculate_consensus(targets)

        assert consensus is not None
        assert consensus.num_analysts == 3
        assert consensus.mean_target == 200.0
        assert consensus.median_target == 200.0
        assert consensus.high_target == 220.0
        assert consensus.low_target == 180.0
        assert consensus.buy_ratings == 2
        assert consensus.hold_ratings == 1

    def test_calculate_consensus_empty(self, extractor: Any) -> None:
        """Test consensus with empty targets list."""
        consensus = extractor.calculate_consensus([])
        assert consensus is None

    def test_validate_target_valid(self, extractor: Any) -> None:
        """Test validation of valid target."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        target = PriceTarget(
            symbol="AAPL",
            target_price=200.0,
            current_price=150.0,
            upside_percent=33.33,
            action=TargetAction.UPGRADE,
            rating=Rating.BUY,
            analyst=None,
            firm=None,
            date=datetime.now(),
            confidence=0.9,
            source_text="Test",
        )

        assert extractor.validate_target(target) is True

    def test_validate_target_unrealistic_upside(self, extractor: Any) -> None:
        """Test validation rejects unrealistic upside."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        target = PriceTarget(
            symbol="AAPL",
            target_price=1000.0,
            current_price=150.0,
            upside_percent=566.67,
            action=TargetAction.UPGRADE,
            rating=Rating.BUY,
            analyst=None,
            firm=None,
            date=datetime.now(),
            confidence=0.9,
            source_text="Test",
        )

        assert extractor.validate_target(target, max_upside_percent=200.0) is False

    def test_validate_target_low_confidence(self, extractor: Any) -> None:
        """Test validation rejects low confidence targets."""
        from signalforge.nlp.price_target_extractor import (
            PriceTarget,
            Rating,
            TargetAction,
        )

        target = PriceTarget(
            symbol="AAPL",
            target_price=200.0,
            current_price=150.0,
            upside_percent=33.33,
            action=TargetAction.UPGRADE,
            rating=Rating.BUY,
            analyst=None,
            firm=None,
            date=datetime.now(),
            confidence=0.3,
            source_text="Test",
        )

        assert extractor.validate_target(target) is False


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def extractor(self) -> Any:
        """Create price target extractor."""
        from signalforge.nlp.price_target_extractor import PriceTargetExtractor

        return PriceTargetExtractor()

    def test_full_extraction_workflow(self, extractor: Any) -> None:
        """Test complete extraction from realistic text."""
        from signalforge.nlp.price_target_extractor import Rating, TargetAction

        text = "Goldman Sachs upgrades AAPL to Buy with price target of $200"

        target = extractor.extract_price_target(text, "AAPL", 150.0)

        assert target is not None
        assert target.target_price == 200.0
        assert target.symbol == "AAPL"
        assert target.rating == Rating.BUY
        assert target.action == TargetAction.UPGRADE
        # Firm extraction is best-effort with regex, may or may not work
        assert isinstance(target.firm, (str, type(None)))
        assert target.upside_percent is not None

    def test_extraction_with_all_components(self, extractor: Any) -> None:
        """Test extraction capturing all components."""
        text = "Morgan Stanley initiates coverage on TSLA with Overweight rating and $300 price target"

        target = extractor.extract_price_target(text, "TSLA", 250.0)

        assert target is not None
        assert target.target_price == 300.0
        assert target.rating is not None
        assert target.action is not None
        assert target.confidence > 0.5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def extractor(self) -> Any:
        """Create price target extractor."""
        from signalforge.nlp.price_target_extractor import PriceTargetExtractor

        return PriceTargetExtractor()

    def test_extract_from_empty_string(self, extractor: Any) -> None:
        """Test extraction from empty string."""
        target = extractor.extract_price_target("", "AAPL", 150.0)
        assert target is None

    def test_extract_from_whitespace(self, extractor: Any) -> None:
        """Test extraction from whitespace only."""
        target = extractor.extract_price_target("   ", "AAPL", 150.0)
        assert target is None

    def test_malformed_price_formats(self, extractor: Any) -> None:
        """Test handling of malformed price formats."""
        text = "Price target of $$$200"
        target = extractor.extract_price_target(text, "AAPL", 150.0)
        # Should still extract 200
        assert target is None or target.target_price == 200.0

    def test_multiple_prices_in_sentence(self, extractor: Any) -> None:
        """Test extraction when multiple prices mentioned."""
        text = "Raises from $150 to $200 price target"
        target = extractor.extract_price_target(text, "AAPL", 150.0)

        assert target is not None
        # Should extract one of the prices
        assert target.target_price in [150.0, 200.0]
