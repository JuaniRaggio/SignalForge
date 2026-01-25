"""Slippage estimation for order execution.

This module provides realistic slippage estimation for trading orders based on
market microstructure theory and empirical observations.

Key Features:
- Order size relative to Average Daily Volume (ADV) analysis
- Volatility-adjusted slippage estimation
- Execution risk classification (low, medium, high)
- Market impact modeling

The slippage model uses a square-root impact function, which is well-established
in market microstructure literature and reflects diminishing marginal impact.

Examples:
    Basic slippage estimation:

    >>> from signalforge.execution.slippage import estimate_slippage
    >>>
    >>> # Order for $50,000 in a stock with $10M average daily volume
    >>> estimate = estimate_slippage(
    ...     order_size_usd=50_000,
    ...     avg_daily_volume=5_000_000,  # shares
    ...     current_price=100.0,
    ...     volatility=0.02,  # 2% daily volatility
    ... )
    >>> print(f"Estimated slippage: {estimate.estimated_slippage_pct:.4f}%")
    >>> print(f"Execution risk: {estimate.execution_risk}")

    Risk categorization:

    >>> from signalforge.execution.slippage import calculate_execution_risk
    >>>
    >>> # Order is 2% of ADV
    >>> risk = calculate_execution_risk(adv_ratio=0.02)
    >>> print(f"Risk level: {risk}")  # "medium"
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from signalforge.core.logging import get_logger

logger = get_logger(__name__)

# Slippage model parameters
BASE_IMPACT_BPS = 10.0  # 10 basis points (0.1%)
REFERENCE_VOLATILITY = 0.02  # 2% daily volatility (reference point)

# Execution risk thresholds (as fraction of ADV)
LOW_RISK_THRESHOLD = 0.01  # 1% of ADV
MEDIUM_RISK_THRESHOLD = 0.05  # 5% of ADV


@dataclass
class SlippageEstimate:
    """Estimated slippage for an order.

    Attributes:
        symbol: Asset symbol (optional, for tracking).
        order_size: Order size in dollars (notional value).
        estimated_slippage_pct: Estimated slippage as a percentage of order value.
        estimated_slippage_usd: Estimated slippage cost in dollars.
        adv_ratio: Order size relative to Average Daily Volume (order_size / ADV).
        execution_risk: Risk category based on order size (low, medium, high).
    """

    symbol: str
    order_size: float
    estimated_slippage_pct: float
    estimated_slippage_usd: float
    adv_ratio: float
    execution_risk: Literal["low", "medium", "high"]

    def __post_init__(self) -> None:
        """Validate slippage estimate."""
        if self.order_size < 0:
            raise ValueError("order_size cannot be negative")
        if self.estimated_slippage_pct < 0:
            raise ValueError("estimated_slippage_pct cannot be negative")
        if self.estimated_slippage_usd < 0:
            raise ValueError("estimated_slippage_usd cannot be negative")
        if self.adv_ratio < 0:
            raise ValueError("adv_ratio cannot be negative")
        if self.execution_risk not in ("low", "medium", "high"):
            raise ValueError("execution_risk must be 'low', 'medium', or 'high'")

    def to_dict(self) -> dict[str, str | float]:
        """Convert estimate to dictionary for serialization."""
        return asdict(self)


def calculate_execution_risk(adv_ratio: float) -> Literal["low", "medium", "high"]:
    """Determine execution risk based on order size relative to ADV.

    Risk Classification:
    - Low Risk: Order < 1% of ADV
      * Minimal market impact expected
      * Can likely execute at or near mid-price
      * Suitable for most market conditions

    - Medium Risk: Order 1-5% of ADV
      * Moderate market impact expected
      * May need to use limit orders or VWAP strategies
      * Consider splitting into multiple orders

    - High Risk: Order > 5% of ADV
      * Significant market impact expected
      * Requires sophisticated execution algorithms
      * Consider delaying execution or using dark pools

    Args:
        adv_ratio: Order size as a fraction of Average Daily Volume.

    Returns:
        Risk level as "low", "medium", or "high".

    Raises:
        ValueError: If adv_ratio is negative.

    Examples:
        >>> calculate_execution_risk(0.005)  # 0.5% of ADV
        'low'
        >>> calculate_execution_risk(0.03)  # 3% of ADV
        'medium'
        >>> calculate_execution_risk(0.08)  # 8% of ADV
        'high'
    """
    if adv_ratio < 0:
        raise ValueError("adv_ratio cannot be negative")

    risk: Literal["low", "medium", "high"]
    if adv_ratio < LOW_RISK_THRESHOLD:
        risk = "low"
    elif adv_ratio < MEDIUM_RISK_THRESHOLD:
        risk = "medium"
    else:
        risk = "high"

    logger.debug(
        "calculated_execution_risk",
        adv_ratio=adv_ratio,
        risk=risk,
    )

    return risk


def estimate_slippage(
    order_size_usd: float,
    avg_daily_volume: float,
    current_price: float,
    volatility: float = REFERENCE_VOLATILITY,
    symbol: str = "",
) -> SlippageEstimate:
    """Estimate slippage based on order size relative to ADV.

    This function implements a square-root market impact model, which is
    well-documented in academic literature (e.g., Almgren & Chriss, 2000).

    Formula:
    --------
    slippage_pct = base_impact * sqrt(order_size / ADV) * volatility_factor

    Where:
    - base_impact = 0.1% (10 basis points)
    - volatility_factor = current_volatility / reference_volatility (2%)
    - Square-root function reflects diminishing marginal impact

    Assumptions:
    -----------
    - Liquid market with continuous trading
    - Normal market conditions (not during news events)
    - Market orders executed at prevailing market price
    - No algorithmic execution strategies applied

    Args:
        order_size_usd: Order size in dollars (notional value).
        avg_daily_volume: Average daily volume in shares/units.
        current_price: Current price of the asset in dollars.
        volatility: Expected daily volatility (default: 2%).
        symbol: Asset symbol for tracking (optional).

    Returns:
        SlippageEstimate object with detailed cost breakdown.

    Raises:
        ValueError: If inputs are invalid (negative values, zero price/volume).

    Examples:
        >>> # Small order in liquid stock
        >>> estimate = estimate_slippage(
        ...     order_size_usd=10_000,
        ...     avg_daily_volume=10_000_000,
        ...     current_price=50.0,
        ...     volatility=0.015,
        ... )
        >>> print(f"Slippage: ${estimate.estimated_slippage_usd:.2f}")

        >>> # Large order in less liquid stock
        >>> estimate = estimate_slippage(
        ...     order_size_usd=100_000,
        ...     avg_daily_volume=1_000_000,
        ...     current_price=25.0,
        ...     volatility=0.03,
        ... )
        >>> print(f"Risk: {estimate.execution_risk}")
    """
    # Input validation
    if order_size_usd < 0:
        raise ValueError("order_size_usd cannot be negative")
    if avg_daily_volume <= 0:
        raise ValueError("avg_daily_volume must be positive")
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")

    logger.info(
        "estimating_slippage",
        symbol=symbol,
        order_size_usd=order_size_usd,
        avg_daily_volume=avg_daily_volume,
        current_price=current_price,
        volatility=volatility,
    )

    # Handle edge case of zero order size
    if order_size_usd == 0:
        logger.debug("estimate_slippage called with zero order size")
        return SlippageEstimate(
            symbol=symbol,
            order_size=0.0,
            estimated_slippage_pct=0.0,
            estimated_slippage_usd=0.0,
            adv_ratio=0.0,
            execution_risk="low",
        )

    # Calculate ADV dollar volume
    adv_dollar_volume = avg_daily_volume * current_price

    # Calculate order size relative to ADV (in dollar terms)
    adv_ratio = order_size_usd / adv_dollar_volume

    # Calculate volatility adjustment factor
    volatility_factor = volatility / REFERENCE_VOLATILITY

    # Calculate slippage using square-root impact model
    # Square root of ADV ratio captures diminishing marginal impact
    base_impact_pct = BASE_IMPACT_BPS / 10000.0  # Convert bps to percentage

    if adv_ratio > 0:
        slippage_pct = base_impact_pct * math.sqrt(adv_ratio) * volatility_factor
    else:
        slippage_pct = 0.0

    # Convert to percentage
    slippage_pct_display = slippage_pct * 100.0

    # Calculate dollar cost
    slippage_usd = order_size_usd * slippage_pct

    # Determine execution risk
    execution_risk = calculate_execution_risk(adv_ratio)

    estimate = SlippageEstimate(
        symbol=symbol,
        order_size=order_size_usd,
        estimated_slippage_pct=slippage_pct_display,
        estimated_slippage_usd=slippage_usd,
        adv_ratio=adv_ratio,
        execution_risk=execution_risk,
    )

    logger.info(
        "slippage_estimated",
        symbol=symbol,
        order_size_usd=order_size_usd,
        slippage_pct=slippage_pct_display,
        slippage_usd=slippage_usd,
        adv_ratio=adv_ratio,
        execution_risk=execution_risk,
    )

    return estimate
