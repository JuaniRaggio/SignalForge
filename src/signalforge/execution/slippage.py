"""Slippage estimation for market impact and execution costs.

This module implements market impact models to estimate slippage costs
for trade execution based on order size, liquidity, and market conditions.
"""

from __future__ import annotations

import math

from signalforge.core.logging import get_logger
from signalforge.execution.schemas import SlippageEstimate

logger = get_logger(__name__)


class SlippageEstimator:
    """Estimate market impact and slippage for trade orders.

    Uses a square-root market impact model which is well-established in
    academic literature and reflects diminishing marginal impact as order
    size increases.

    The model estimates slippage as:
        Impact = coefficient * sigma * sqrt(order_size / ADV)

    Where:
        - coefficient: Market impact coefficient (default 0.1)
        - sigma: Asset volatility
        - order_size: Order size relative to ADV

    References:
        Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio
        transactions. Journal of Risk.
    """

    def __init__(self, impact_coefficient: float = 0.1) -> None:
        """Initialize slippage estimator.

        Args:
            impact_coefficient: Market impact coefficient (default: 0.1).
                Higher values indicate greater market impact.

        Raises:
            ValueError: If coefficient is negative.
        """
        if impact_coefficient < 0:
            raise ValueError("Impact coefficient must be non-negative")

        self.impact_coefficient = impact_coefficient

        logger.debug("slippage_estimator_initialized", impact_coefficient=impact_coefficient)

    def estimate(
        self,
        symbol: str,
        order_size_shares: int,
        adv_20: float,
        avg_spread_bps: float,
        volatility: float | None = None,
    ) -> SlippageEstimate:
        """Estimate slippage using square-root market impact model.

        Formula:
            slippage_pct = base_spread + impact_coefficient * sigma * sqrt(Q / ADV)

        Where:
            - base_spread: Half the bid-ask spread (in bps)
            - sigma: Daily volatility (default 2% if not provided)
            - Q: Order size in shares
            - ADV: Average daily volume

        Args:
            symbol: Trading symbol.
            order_size_shares: Order size in shares.
            adv_20: Average daily volume over 20 days.
            avg_spread_bps: Average bid-ask spread in basis points.
            volatility: Daily volatility (optional, default 0.02 = 2%).

        Returns:
            SlippageEstimate with cost breakdown.

        Raises:
            ValueError: If inputs are invalid.
        """
        if order_size_shares < 0:
            raise ValueError("Order size cannot be negative")
        if adv_20 <= 0:
            raise ValueError("Average daily volume must be positive")
        if avg_spread_bps < 0:
            raise ValueError("Spread cannot be negative")

        logger.info(
            "estimating_slippage",
            symbol=symbol,
            order_size_shares=order_size_shares,
            adv_20=adv_20,
            avg_spread_bps=avg_spread_bps,
            volatility=volatility,
        )

        # Default volatility if not provided
        if volatility is None:
            volatility = 0.02  # 2% daily volatility

        # Handle zero order size edge case
        if order_size_shares == 0:
            return SlippageEstimate(
                symbol=symbol,
                order_size=0.0,
                estimated_slippage_bps=0.0,
                estimated_slippage_pct=0.0,
                confidence=1.0,
                market_impact_cost=0.0,
            )

        # Calculate order size as fraction of ADV
        adv_ratio = order_size_shares / adv_20

        # Base cost from crossing the spread (half spread)
        base_cost_bps = avg_spread_bps / 2.0

        # Market impact using square-root model
        # Impact scales with sqrt of relative order size and volatility
        if adv_ratio > 0:
            impact_bps = (
                self.impact_coefficient * (volatility * 100) * math.sqrt(adv_ratio) * 100
            )
        else:
            impact_bps = 0.0

        # Total slippage = base cost + market impact
        total_slippage_bps = base_cost_bps + impact_bps

        # Convert to percentage
        slippage_pct = total_slippage_bps / 100.0

        # Estimate confidence based on order size relative to ADV
        # High confidence for small orders, lower for large orders
        if adv_ratio < 0.01:  # < 1% of ADV
            confidence = 0.95
        elif adv_ratio < 0.05:  # < 5% of ADV
            confidence = 0.80
        elif adv_ratio < 0.10:  # < 10% of ADV
            confidence = 0.60
        else:  # >= 10% of ADV
            confidence = 0.40

        # Calculate dollar cost (assumes we know the order value)
        # For now, just use the slippage percentage
        market_impact_cost = 0.0  # Will be calculated when we have order dollar value

        estimate = SlippageEstimate(
            symbol=symbol,
            order_size=float(order_size_shares),
            estimated_slippage_bps=total_slippage_bps,
            estimated_slippage_pct=slippage_pct,
            confidence=confidence,
            market_impact_cost=market_impact_cost,
        )

        logger.info(
            "slippage_estimated",
            symbol=symbol,
            slippage_bps=total_slippage_bps,
            slippage_pct=slippage_pct,
            confidence=confidence,
            adv_ratio=adv_ratio,
        )

        return estimate

    def estimate_for_dollar_amount(
        self,
        symbol: str,
        dollar_amount: float,
        price: float,
        adv_20: float,
        avg_spread_bps: float,
    ) -> SlippageEstimate:
        """Estimate slippage for dollar-denominated order.

        Args:
            symbol: Trading symbol.
            dollar_amount: Order size in dollars.
            price: Current share price.
            adv_20: Average daily volume in shares.
            avg_spread_bps: Average spread in basis points.

        Returns:
            SlippageEstimate with cost breakdown including dollar impact.

        Raises:
            ValueError: If inputs are invalid.
        """
        if dollar_amount < 0:
            raise ValueError("Dollar amount cannot be negative")
        if price <= 0:
            raise ValueError("Price must be positive")
        if adv_20 <= 0:
            raise ValueError("Average daily volume must be positive")

        # Convert dollar amount to shares
        order_size_shares = int(dollar_amount / price)

        # Get base estimate
        estimate = self.estimate(
            symbol=symbol,
            order_size_shares=order_size_shares,
            adv_20=adv_20,
            avg_spread_bps=avg_spread_bps,
        )

        # Calculate market impact cost in dollars
        market_impact_cost = dollar_amount * (estimate.estimated_slippage_pct / 100.0)

        # Update estimate with dollar values
        return SlippageEstimate(
            symbol=estimate.symbol,
            order_size=dollar_amount,
            estimated_slippage_bps=estimate.estimated_slippage_bps,
            estimated_slippage_pct=estimate.estimated_slippage_pct,
            confidence=estimate.confidence,
            market_impact_cost=market_impact_cost,
        )
