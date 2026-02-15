"""Position sizing module using Kelly criterion and risk constraints.

This module implements position sizing strategies for portfolio management,
including Kelly criterion calculations with fractional sizing and comprehensive
portfolio constraints.

The Kelly criterion formula: f* = (p * b - q) / b
where:
    p = win_rate (probability of winning)
    q = 1 - p (probability of losing)
    b = avg_win / avg_loss (win/loss ratio)
    f* = optimal fraction of capital to risk

Examples:
    Calculate position size with Kelly criterion:

    >>> from decimal import Decimal
    >>> from signalforge.risk import PositionSizer, PositionSizeConfig
    >>>
    >>> config = PositionSizeConfig(
    ...     max_position_pct=0.1,
    ...     kelly_fraction=0.5,
    ...     min_position_size=Decimal("100")
    ... )
    >>> sizer = PositionSizer(config)
    >>> result = sizer.calculate_position_size(
    ...     portfolio_value=Decimal("100000"),
    ...     price=Decimal("50.00"),
    ...     win_rate=0.55,
    ...     avg_win=Decimal("100"),
    ...     avg_loss=Decimal("50")
    ... )
    >>> print(f"Shares: {result.shares}, Amount: ${result.dollar_amount}")
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSizeConfig:
    """Configuration for position sizing constraints.

    Attributes:
        max_position_pct: Maximum percentage of portfolio for a single position (default: 0.1)
        max_portfolio_pct: Maximum total portfolio allocation percentage (default: 0.25)
        kelly_fraction: Fraction of full Kelly to use for safety (default: 0.5)
        min_position_size: Minimum position size in currency units

    Raises:
        ValueError: If any percentage is not in valid range or min_position_size is negative
    """

    max_position_pct: float = 0.1
    max_portfolio_pct: float = 0.25
    kelly_fraction: float = 0.5
    min_position_size: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0 < self.max_position_pct <= 1:
            raise ValueError(
                f"max_position_pct must be between 0 and 1, got {self.max_position_pct}"
            )
        if not 0 < self.max_portfolio_pct <= 1:
            raise ValueError(
                f"max_portfolio_pct must be between 0 and 1, got {self.max_portfolio_pct}"
            )
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError(
                f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}"
            )
        if self.min_position_size < 0:
            raise ValueError(
                f"min_position_size cannot be negative, got {self.min_position_size}"
            )
        if self.max_position_pct > self.max_portfolio_pct:
            logger.warning(
                "max_position_pct exceeds max_portfolio_pct",
                max_position_pct=self.max_position_pct,
                max_portfolio_pct=self.max_portfolio_pct,
            )


@dataclass
class PositionSizeResult:
    """Result of position size calculation.

    Attributes:
        shares: Number of shares to purchase (integer)
        dollar_amount: Total dollar amount of position
        portfolio_pct: Percentage of portfolio this position represents
        kelly_full: Full Kelly criterion fraction
        kelly_adjusted: Kelly fraction after applying kelly_fraction multiplier
        risk_amount: Amount of capital at risk for this position

    Raises:
        ValueError: If shares is negative or portfolio_pct is invalid
    """

    shares: int
    dollar_amount: Decimal
    portfolio_pct: float
    kelly_full: float
    kelly_adjusted: float
    risk_amount: Decimal

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if self.shares < 0:
            raise ValueError(f"shares cannot be negative, got {self.shares}")
        if self.dollar_amount < 0:
            raise ValueError(f"dollar_amount cannot be negative, got {self.dollar_amount}")
        if not 0 <= self.portfolio_pct <= 1:
            raise ValueError(
                f"portfolio_pct must be between 0 and 1, got {self.portfolio_pct}"
            )
        if self.risk_amount < 0:
            raise ValueError(f"risk_amount cannot be negative, got {self.risk_amount}")


class PositionSizer:
    """Position sizing calculator using Kelly criterion with constraints.

    This class calculates optimal position sizes based on the Kelly criterion,
    applying fractional sizing and portfolio constraints for risk management.

    Attributes:
        config: Position sizing configuration

    Examples:
        Initialize and calculate position size:

        >>> from decimal import Decimal
        >>> config = PositionSizeConfig(max_position_pct=0.15, kelly_fraction=0.5)
        >>> sizer = PositionSizer(config)
        >>> result = sizer.calculate_position_size(
        ...     portfolio_value=Decimal("100000"),
        ...     price=Decimal("100"),
        ...     win_rate=0.6,
        ...     avg_win=Decimal("200"),
        ...     avg_loss=Decimal("100")
        ... )
    """

    def __init__(self, config: PositionSizeConfig) -> None:
        """Initialize the position sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config
        logger.info(
            "position_sizer_initialized",
            max_position_pct=config.max_position_pct,
            max_portfolio_pct=config.max_portfolio_pct,
            kelly_fraction=config.kelly_fraction,
            min_position_size=str(config.min_position_size),
        )

    def calculate_kelly(
        self, win_rate: float, avg_win: Decimal, avg_loss: Decimal
    ) -> float:
        """Calculate full Kelly criterion fraction.

        Formula: f* = (p * b - q) / b
        where p = win_rate, q = 1 - p, b = avg_win / avg_loss

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)

        Returns:
            Kelly fraction (can be negative if expected value is negative)

        Raises:
            ValueError: If win_rate is not in [0, 1] or avg_win/avg_loss are negative

        Examples:
            >>> from decimal import Decimal
            >>> sizer = PositionSizer(PositionSizeConfig())
            >>> kelly = sizer.calculate_kelly(0.6, Decimal("100"), Decimal("50"))
            >>> print(f"Kelly fraction: {kelly:.2%}")
        """
        if not 0 <= win_rate <= 1:
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_win < 0:
            raise ValueError(f"avg_win cannot be negative, got {avg_win}")
        if avg_loss < 0:
            raise ValueError(f"avg_loss cannot be negative, got {avg_loss}")

        # Handle edge cases
        if win_rate == 0:
            logger.debug("calculate_kelly: win_rate is zero, returning -1.0")
            return -1.0

        if win_rate == 1:
            logger.debug("calculate_kelly: win_rate is 1.0, returning 1.0")
            return 1.0

        if avg_loss == 0:
            if avg_win > 0 and win_rate > 0:
                logger.debug("calculate_kelly: avg_loss is zero with positive avg_win")
                return 1.0
            logger.debug("calculate_kelly: avg_loss is zero, returning 0.0")
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        p = win_rate
        q = 1.0 - win_rate
        b = float(avg_win / avg_loss)

        kelly = (p * b - q) / b

        logger.debug(
            "kelly_calculated",
            win_rate=win_rate,
            avg_win=str(avg_win),
            avg_loss=str(avg_loss),
            b_ratio=b,
            kelly=kelly,
        )

        return kelly

    def calculate_half_kelly(
        self, win_rate: float, avg_win: Decimal, avg_loss: Decimal
    ) -> float:
        """Calculate half-Kelly fraction (0.5 * Kelly).

        Half-Kelly is a common conservative approach that reduces volatility
        while maintaining most of the growth potential.

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)

        Returns:
            Half-Kelly fraction

        Examples:
            >>> from decimal import Decimal
            >>> sizer = PositionSizer(PositionSizeConfig())
            >>> half_kelly = sizer.calculate_half_kelly(0.6, Decimal("100"), Decimal("50"))
        """
        return self.calculate_fractional_kelly(win_rate, avg_win, avg_loss, 0.5)

    def calculate_fractional_kelly(
        self, win_rate: float, avg_win: Decimal, avg_loss: Decimal, fraction: float
    ) -> float:
        """Calculate fractional Kelly criterion.

        Fractional Kelly reduces risk by using a fraction of the full Kelly.
        Common fractions are 0.25 (quarter-Kelly) and 0.5 (half-Kelly).

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            fraction: Fraction of full Kelly to use (0-1)

        Returns:
            Fractional Kelly value

        Raises:
            ValueError: If fraction is not in (0, 1]

        Examples:
            >>> from decimal import Decimal
            >>> sizer = PositionSizer(PositionSizeConfig())
            >>> # Quarter-Kelly for very conservative sizing
            >>> quarter = sizer.calculate_fractional_kelly(
            ...     0.6, Decimal("100"), Decimal("50"), 0.25
            ... )
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

        full_kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)
        fractional = full_kelly * fraction

        logger.debug(
            "fractional_kelly_calculated",
            full_kelly=full_kelly,
            fraction=fraction,
            fractional_kelly=fractional,
        )

        return fractional

    def calculate_position_size(
        self,
        portfolio_value: Decimal,
        price: Decimal,
        win_rate: float,
        avg_win: Decimal,
        avg_loss: Decimal,
    ) -> PositionSizeResult:
        """Calculate optimal position size with Kelly criterion and constraints.

        This is the main method that combines Kelly calculation with portfolio
        constraints to determine the actual position size.

        Args:
            portfolio_value: Total portfolio value
            price: Price per share
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)

        Returns:
            PositionSizeResult with calculated position details

        Raises:
            ValueError: If portfolio_value or price are not positive

        Examples:
            >>> from decimal import Decimal
            >>> sizer = PositionSizer(PositionSizeConfig())
            >>> result = sizer.calculate_position_size(
            ...     portfolio_value=Decimal("100000"),
            ...     price=Decimal("50"),
            ...     win_rate=0.55,
            ...     avg_win=Decimal("100"),
            ...     avg_loss=Decimal("80")
            ... )
            >>> print(f"Buy {result.shares} shares at ${price}")
        """
        if portfolio_value <= 0:
            raise ValueError(f"portfolio_value must be positive, got {portfolio_value}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        # Calculate Kelly fractions
        kelly_full = self.calculate_kelly(win_rate, avg_win, avg_loss)
        kelly_adjusted = kelly_full * self.config.kelly_fraction

        # If Kelly is negative or zero, no position
        if kelly_adjusted <= 0:
            logger.info(
                "position_size_zero_negative_kelly",
                kelly_full=kelly_full,
                kelly_adjusted=kelly_adjusted,
            )
            return PositionSizeResult(
                shares=0,
                dollar_amount=Decimal("0"),
                portfolio_pct=0.0,
                kelly_full=kelly_full,
                kelly_adjusted=kelly_adjusted,
                risk_amount=Decimal("0"),
            )

        # Calculate dollar amount based on Kelly
        dollar_amount = portfolio_value * Decimal(str(kelly_adjusted))

        # Calculate shares (floor to get integer shares)
        shares = int(dollar_amount / price)
        actual_dollar_amount = Decimal(shares) * price

        # Calculate portfolio percentage
        portfolio_pct = float(actual_dollar_amount / portfolio_value)

        # Estimate risk amount (using avg_loss as proxy)
        risk_amount = actual_dollar_amount * (avg_loss / (avg_win + avg_loss))

        result = PositionSizeResult(
            shares=shares,
            dollar_amount=actual_dollar_amount,
            portfolio_pct=portfolio_pct,
            kelly_full=kelly_full,
            kelly_adjusted=kelly_adjusted,
            risk_amount=risk_amount,
        )

        # Apply constraints
        constrained_result = self.apply_constraints(result, portfolio_value)

        logger.info(
            "position_size_calculated",
            original_shares=shares,
            constrained_shares=constrained_result.shares,
            portfolio_pct=constrained_result.portfolio_pct,
            kelly_full=kelly_full,
            kelly_adjusted=kelly_adjusted,
        )

        return constrained_result

    def apply_constraints(
        self, result: PositionSizeResult, portfolio_value: Decimal
    ) -> PositionSizeResult:
        """Apply portfolio constraints to position size.

        Constraints enforced:
        1. Position size cannot exceed max_position_pct of portfolio
        2. Position size cannot exceed max_portfolio_pct of total allocation
        3. Position size must be >= min_position_size or zero

        Args:
            result: Original position size result
            portfolio_value: Total portfolio value

        Returns:
            Constrained PositionSizeResult

        Examples:
            >>> from decimal import Decimal
            >>> config = PositionSizeConfig(max_position_pct=0.1)
            >>> sizer = PositionSizer(config)
            >>> # This would apply constraints to an oversized position
        """
        shares = result.shares
        dollar_amount = result.dollar_amount

        # Constraint 1: Max position percentage
        max_position_dollars = portfolio_value * Decimal(str(self.config.max_position_pct))
        if dollar_amount > max_position_dollars:
            logger.debug(
                "applying_max_position_constraint",
                original_amount=str(dollar_amount),
                max_allowed=str(max_position_dollars),
            )
            dollar_amount = max_position_dollars
            # Recalculate shares if we have the price
            if result.shares > 0:
                price = result.dollar_amount / Decimal(result.shares)
                shares = int(dollar_amount / price)
                dollar_amount = Decimal(shares) * price

        # Constraint 2: Max portfolio percentage (similar to max_position in this context)
        max_portfolio_dollars = portfolio_value * Decimal(str(self.config.max_portfolio_pct))
        if dollar_amount > max_portfolio_dollars:
            logger.debug(
                "applying_max_portfolio_constraint",
                original_amount=str(dollar_amount),
                max_allowed=str(max_portfolio_dollars),
            )
            dollar_amount = max_portfolio_dollars
            if result.shares > 0:
                price = result.dollar_amount / Decimal(result.shares)
                shares = int(dollar_amount / price)
                dollar_amount = Decimal(shares) * price

        # Constraint 3: Minimum position size
        if dollar_amount > 0 and dollar_amount < self.config.min_position_size:
            logger.debug(
                "position_below_minimum",
                dollar_amount=str(dollar_amount),
                min_position_size=str(self.config.min_position_size),
            )
            shares = 0
            dollar_amount = Decimal("0")

        # Recalculate metrics
        portfolio_pct = float(dollar_amount / portfolio_value) if portfolio_value > 0 else 0.0

        # Recalculate risk amount proportionally
        if result.dollar_amount > 0:
            risk_ratio = dollar_amount / result.dollar_amount
            risk_amount = result.risk_amount * risk_ratio
        else:
            risk_amount = Decimal("0")

        return PositionSizeResult(
            shares=shares,
            dollar_amount=dollar_amount,
            portfolio_pct=portfolio_pct,
            kelly_full=result.kelly_full,
            kelly_adjusted=result.kelly_adjusted,
            risk_amount=risk_amount,
        )
