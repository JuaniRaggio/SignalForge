"""Extract price targets from analyst reports and news.

This module provides functionality to extract and aggregate price targets,
analyst ratings, and price target actions from financial text using regex
patterns and heuristics.

Key Features:
- Price target extraction from text
- Analyst rating classification
- Action detection (upgrade, downgrade, maintain, etc.)
- Analyst and firm extraction
- Consensus target calculation
- Target validation

Examples:
    Extract a price target:

    >>> from signalforge.nlp.price_target_extractor import PriceTargetExtractor
    >>>
    >>> extractor = PriceTargetExtractor()
    >>> text = "Morgan Stanley raises AAPL price target to $200"
    >>> target = extractor.extract_price_target(text, "AAPL", 150.0)
    >>> print(f"Target: ${target.target_price}, Upside: {target.upside_percent}%")
    Target: $200, Upside: 33.3%

    Calculate consensus from multiple targets:

    >>> targets = [...]  # List of PriceTarget objects
    >>> consensus = extractor.calculate_consensus(targets)
    >>> print(f"Mean: ${consensus.mean_target}")
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


class TargetAction(str, Enum):
    """Price target action classification."""

    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    MAINTAIN = "maintain"
    INITIATE = "initiate"
    REITERATE = "reiterate"


class Rating(str, Enum):
    """Analyst rating classification."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class PriceTarget:
    """Extracted price target from analyst report.

    Attributes:
        symbol: Stock ticker symbol.
        target_price: Price target value.
        current_price: Current stock price (if available).
        upside_percent: Percentage upside to target (if current_price available).
        action: Type of action (upgrade, downgrade, etc.).
        rating: Analyst rating (buy, hold, sell, etc.).
        analyst: Analyst name (if extracted).
        firm: Analyst firm name (if extracted).
        date: Date of the price target.
        confidence: Extraction confidence score (0.0 to 1.0).
        source_text: Original text containing the price target.
    """

    symbol: str
    target_price: float
    current_price: float | None
    upside_percent: float | None
    action: TargetAction
    rating: Rating | None
    analyst: str | None
    firm: str | None
    date: datetime
    confidence: float
    source_text: str

    def __post_init__(self) -> None:
        """Validate price target fields."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if self.target_price <= 0:
            raise ValueError(f"target_price must be positive, got {self.target_price}")

        if self.current_price is not None and self.current_price < 0:
            raise ValueError(f"current_price must be non-negative, got {self.current_price}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Calculate upside if not provided
        if self.upside_percent is None and self.current_price is not None and self.current_price > 0:
            self.upside_percent = (
                (self.target_price - self.current_price) / self.current_price * 100
            )


@dataclass
class ConsensusTarget:
    """Consensus price target from multiple analysts.

    Attributes:
        symbol: Stock ticker symbol.
        mean_target: Mean of all price targets.
        median_target: Median of all price targets.
        high_target: Highest price target.
        low_target: Lowest price target.
        num_analysts: Number of analysts contributing.
        buy_ratings: Count of buy/strong buy ratings.
        hold_ratings: Count of hold/neutral ratings.
        sell_ratings: Count of sell/strong sell ratings.
        consensus_rating: Overall consensus rating.
    """

    symbol: str
    mean_target: float
    median_target: float
    high_target: float
    low_target: float
    num_analysts: int
    buy_ratings: int
    hold_ratings: int
    sell_ratings: int
    consensus_rating: Rating

    def __post_init__(self) -> None:
        """Validate consensus target fields."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if self.num_analysts <= 0:
            raise ValueError(f"num_analysts must be positive, got {self.num_analysts}")

        if self.mean_target <= 0:
            raise ValueError(f"mean_target must be positive, got {self.mean_target}")


class PriceTargetExtractor:
    """Extract and aggregate price targets from text.

    This extractor uses regex patterns to identify price targets, analyst
    ratings, and related information from financial news and analyst reports.

    Examples:
        >>> extractor = PriceTargetExtractor()
        >>> text = "Wells Fargo upgrades TSLA to Buy with $250 price target"
        >>> target = extractor.extract_price_target(text, "TSLA", 200.0)
        >>> target.action
        <TargetAction.UPGRADE: 'upgrade'>
    """

    # Regex patterns for price extraction
    PRICE_PATTERNS = [
        r"price target (?:of |to |at )?\$?([\d,]+(?:\.\d{2})?)",
        r"target (?:price )?(?:of |to |at )?\$?([\d,]+(?:\.\d{2})?)",
        r"\$?([\d,]+(?:\.\d{2})?) (?:price )?target",
        r"(?:raises?|lowers?|sets?) (?:price )?target (?:to |at )?\$?([\d,]+(?:\.\d{2})?)",
        r"(?:PT|pt) (?:of |to |at )?\$?([\d,]+(?:\.\d{2})?)",
    ]

    RATING_PATTERNS = {
        Rating.STRONG_BUY: [r"strong buy", r"conviction buy", r"top pick", r"strong-buy"],
        Rating.BUY: [
            r"\bbuy\b",
            r"outperform",
            r"overweight",
            r"accumulate",
            r"out-perform",
            r"over-weight",
        ],
        Rating.HOLD: [
            r"\bhold\b",
            r"neutral",
            r"equal.?weight",
            r"market perform",
            r"peer perform",
            r"sector perform",
        ],
        Rating.SELL: [
            r"\bsell\b",
            r"underperform",
            r"underweight",
            r"reduce",
            r"under-perform",
            r"under-weight",
        ],
        Rating.STRONG_SELL: [r"strong sell", r"conviction sell", r"strong-sell"],
    }

    ACTION_PATTERNS = {
        TargetAction.UPGRADE: [r"\bupgrade[sd]?\b", r"raises?", r"increased?", r"ups?"],
        TargetAction.DOWNGRADE: [
            r"\bdowngrade[sd]?\b",
            r"lowers?",
            r"cuts?",
            r"reduced?",
            r"decreased?",
        ],
        TargetAction.INITIATE: [r"initiate[sd]?", r"begins?", r"starts?", r"launches?"],
        TargetAction.REITERATE: [r"reiterate[sd]?", r"reaffirms?", r"maintains?", r"keeps?"],
        TargetAction.MAINTAIN: [r"maintains?", r"unchanged", r"keeps?"],
    }

    ANALYST_FIRM_PATTERNS = [
        r"(?:analyst|strategist)\s+(?:at|from)\s+([A-Z][A-Za-z\s&]+?(?:LLC|Inc|Group|Capital|Securities|Advisors|Partners|Bank|\b))",
        r"([A-Z][A-Za-z\s&]+?(?:LLC|Inc|Group|Capital|Securities|Advisors|Partners|Bank))\s+(?:analyst|strategist|upgrades|downgrades|raises|lowers|initiates)",
    ]

    def __init__(self) -> None:
        """Initialize the price target extractor."""
        self.logger = get_logger(__name__)

        # Compile regex patterns
        self._price_regexes = [re.compile(p, re.IGNORECASE) for p in self.PRICE_PATTERNS]
        self._rating_regexes = {
            rating: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rating, patterns in self.RATING_PATTERNS.items()
        }
        self._action_regexes = {
            action: [re.compile(p, re.IGNORECASE) for p in patterns]
            for action, patterns in self.ACTION_PATTERNS.items()
        }
        self._firm_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.ANALYST_FIRM_PATTERNS
        ]

        self.logger.info("price_target_extractor_initialized")

    def _parse_price(self, price_str: str) -> float | None:
        """Parse price string to float.

        Args:
            price_str: Price string (may contain commas).

        Returns:
            Parsed price as float, or None if parsing fails.
        """
        try:
            # Remove commas and parse
            clean_price = price_str.replace(",", "")
            return float(clean_price)
        except (ValueError, AttributeError):
            return None

    def extract_price_target(
        self,
        text: str,
        symbol: str | None = None,
        current_price: float | None = None,
    ) -> PriceTarget | None:
        """Extract price target from text.

        Args:
            text: Text to extract price target from.
            symbol: Stock symbol (optional, will try to extract if None).
            current_price: Current stock price for upside calculation.

        Returns:
            PriceTarget if found, None otherwise.

        Examples:
            >>> extractor = PriceTargetExtractor()
            >>> text = "Goldman Sachs sets AAPL price target at $195"
            >>> target = extractor.extract_price_target(text, "AAPL", 150.0)
            >>> target.target_price
            195.0
        """
        if not text or not text.strip():
            return None

        # Extract price
        target_price = None
        matched_pattern = None

        for regex in self._price_regexes:
            match = regex.search(text)
            if match:
                price_str = match.group(1)
                target_price = self._parse_price(price_str)
                if target_price is not None:
                    matched_pattern = regex.pattern
                    break

        if target_price is None:
            return None

        # Extract symbol if not provided
        if symbol is None:
            # Simple symbol extraction: look for $SYMBOL or all caps 2-5 letter words
            symbol_match = re.search(r"\$([A-Z]{1,5})\b|^([A-Z]{2,5})\b", text)
            symbol = symbol_match.group(1) or symbol_match.group(2) if symbol_match else "UNKNOWN"

        # Extract rating
        rating = self.extract_rating(text)

        # Extract action
        action = self.extract_action(text)

        # Extract analyst and firm
        analyst, firm = self.extract_analyst_info(text)

        # Calculate confidence based on what we extracted
        confidence = 0.7  # Base confidence
        if rating is not None:
            confidence += 0.1
        if action != TargetAction.MAINTAIN:
            confidence += 0.1
        if firm is not None:
            confidence += 0.1

        confidence = min(confidence, 1.0)

        # Calculate upside if current price available
        upside_percent = None
        if current_price is not None and current_price > 0:
            upside_percent = (target_price - current_price) / current_price * 100

        price_target = PriceTarget(
            symbol=symbol,
            target_price=target_price,
            current_price=current_price,
            upside_percent=upside_percent,
            action=action,
            rating=rating,
            analyst=analyst,
            firm=firm,
            date=datetime.now(UTC),
            confidence=confidence,
            source_text=text[:200],  # Truncate for storage
        )

        self.logger.debug(
            "price_target_extracted",
            symbol=symbol,
            target_price=target_price,
            action=action.value,
            rating=rating.value if rating else None,
            confidence=confidence,
            pattern=matched_pattern,
        )

        return price_target

    def extract_all_targets(self, text: str) -> list[PriceTarget]:
        """Extract all price targets mentioned in text.

        Args:
            text: Text to extract price targets from.

        Returns:
            List of PriceTarget objects found.

        Examples:
            >>> extractor = PriceTargetExtractor()
            >>> text = "AAPL target $200, MSFT target $400"
            >>> targets = extractor.extract_all_targets(text)
            >>> len(targets) >= 1
            True
        """
        if not text or not text.strip():
            return []

        # Split text into sentences
        sentences = re.split(r"[.!?]\s+", text)

        targets = []
        for sentence in sentences:
            target = self.extract_price_target(sentence)
            if target is not None:
                targets.append(target)

        self.logger.debug("all_targets_extracted", num_targets=len(targets))

        return targets

    def extract_rating(self, text: str) -> Rating | None:
        """Extract analyst rating from text.

        Args:
            text: Text to extract rating from.

        Returns:
            Rating if found, None otherwise.

        Examples:
            >>> extractor = PriceTargetExtractor()
            >>> text = "Analyst upgrades to Buy"
            >>> rating = extractor.extract_rating(text)
            >>> rating
            <Rating.BUY: 'buy'>
        """
        if not text:
            return None

        text_lower = text.lower()

        # Check each rating pattern in order of specificity
        for rating in [
            Rating.STRONG_BUY,
            Rating.STRONG_SELL,
            Rating.BUY,
            Rating.SELL,
            Rating.HOLD,
        ]:
            for regex in self._rating_regexes[rating]:
                if regex.search(text_lower):
                    self.logger.debug("rating_extracted", rating=rating.value)
                    return rating

        return None

    def extract_action(self, text: str) -> TargetAction:
        """Determine if this is upgrade/downgrade/etc.

        Args:
            text: Text to extract action from.

        Returns:
            TargetAction classification.

        Examples:
            >>> extractor = PriceTargetExtractor()
            >>> text = "Analyst upgrades stock to Buy"
            >>> action = extractor.extract_action(text)
            >>> action
            <TargetAction.UPGRADE: 'upgrade'>
        """
        if not text:
            return TargetAction.MAINTAIN

        text_lower = text.lower()

        # Check each action pattern
        for action in [
            TargetAction.UPGRADE,
            TargetAction.DOWNGRADE,
            TargetAction.INITIATE,
            TargetAction.REITERATE,
        ]:
            for regex in self._action_regexes[action]:
                if regex.search(text_lower):
                    self.logger.debug("action_extracted", action=action.value)
                    return action

        # Default to maintain if no action found
        return TargetAction.MAINTAIN

    def extract_analyst_info(self, text: str) -> tuple[str | None, str | None]:
        """Extract analyst name and firm.

        Args:
            text: Text to extract analyst information from.

        Returns:
            Tuple of (analyst_name, firm_name).

        Examples:
            >>> extractor = PriceTargetExtractor()
            >>> text = "Analyst at Morgan Stanley raises target"
            >>> analyst, firm = extractor.extract_analyst_info(text)
            >>> firm
            'Morgan Stanley'
        """
        if not text:
            return None, None

        analyst_name = None
        firm_name = None

        # Extract firm
        for regex in self._firm_regexes:
            match = regex.search(text)
            if match:
                firm_name = match.group(1).strip()
                break

        # Analyst name extraction is complex and error-prone with regex
        # For MVP, we'll skip it
        # In production, would use NER (spaCy) to extract PERSON entities

        self.logger.debug("analyst_info_extracted", firm=firm_name)

        return analyst_name, firm_name

    def calculate_consensus(self, targets: list[PriceTarget]) -> ConsensusTarget | None:
        """Calculate consensus from multiple targets.

        Args:
            targets: List of PriceTarget objects for the same symbol.

        Returns:
            ConsensusTarget with aggregated statistics, or None if empty.

        Examples:
            >>> extractor = PriceTargetExtractor()
            >>> targets = [...]  # List of PriceTarget objects
            >>> consensus = extractor.calculate_consensus(targets)
            >>> consensus is not None
            True
        """
        if not targets:
            return None

        # Check all targets are for same symbol
        symbols = {t.symbol for t in targets}
        if len(symbols) > 1:
            self.logger.warning(
                "multiple_symbols_in_consensus",
                symbols=list(symbols),
                message="Targets contain multiple symbols, using first",
            )

        symbol = targets[0].symbol

        # Calculate price statistics
        prices = [t.target_price for t in targets]
        mean_target = statistics.mean(prices)
        median_target = statistics.median(prices)
        high_target = max(prices)
        low_target = min(prices)

        # Count ratings
        buy_ratings = sum(
            1
            for t in targets
            if t.rating in (Rating.BUY, Rating.STRONG_BUY) and t.rating is not None
        )
        hold_ratings = sum(1 for t in targets if t.rating == Rating.HOLD and t.rating is not None)
        sell_ratings = sum(
            1
            for t in targets
            if t.rating in (Rating.SELL, Rating.STRONG_SELL) and t.rating is not None
        )

        # Determine consensus rating
        total_ratings = buy_ratings + hold_ratings + sell_ratings
        if total_ratings == 0:
            consensus_rating = Rating.HOLD
        elif buy_ratings > hold_ratings and buy_ratings > sell_ratings:
            consensus_rating = Rating.BUY
        elif sell_ratings > hold_ratings and sell_ratings > buy_ratings:
            consensus_rating = Rating.SELL
        else:
            consensus_rating = Rating.HOLD

        consensus = ConsensusTarget(
            symbol=symbol,
            mean_target=mean_target,
            median_target=median_target,
            high_target=high_target,
            low_target=low_target,
            num_analysts=len(targets),
            buy_ratings=buy_ratings,
            hold_ratings=hold_ratings,
            sell_ratings=sell_ratings,
            consensus_rating=consensus_rating,
        )

        self.logger.info(
            "consensus_calculated",
            symbol=symbol,
            num_analysts=len(targets),
            mean_target=mean_target,
            consensus_rating=consensus_rating.value,
        )

        return consensus

    def validate_target(
        self, target: PriceTarget, max_upside_percent: float = 200.0
    ) -> bool:
        """Validate extracted target is reasonable.

        Args:
            target: PriceTarget to validate.
            max_upside_percent: Maximum allowed upside percentage.

        Returns:
            True if target is valid, False otherwise.

        Examples:
            >>> extractor = PriceTargetExtractor()
            >>> # Valid target
            >>> target = PriceTarget(...)  # Normal target
            >>> extractor.validate_target(target)
            True
        """
        # Check target price is positive
        if target.target_price <= 0:
            self.logger.warning("invalid_target_price", price=target.target_price)
            return False

        # Check upside is reasonable
        if target.upside_percent is not None and abs(target.upside_percent) > max_upside_percent:
            self.logger.warning(
                "unrealistic_upside",
                upside=target.upside_percent,
                max_allowed=max_upside_percent,
            )
            return False

        # Check confidence threshold
        if target.confidence < 0.5:
            self.logger.warning("low_confidence_target", confidence=target.confidence)
            return False

        return True
