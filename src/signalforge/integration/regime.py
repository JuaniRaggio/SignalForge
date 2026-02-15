"""Market regime detection for signal aggregation."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger
from signalforge.integration.schemas import MarketRegime

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class MarketRegimeDetector:
    """Detect current market regime based on volatility and trend analysis."""

    def __init__(
        self,
        volatility_threshold_high: float = 0.25,
        volatility_threshold_low: float = 0.10,
        trend_threshold: float = 0.05,
    ) -> None:
        """Initialize the market regime detector.

        Args:
            volatility_threshold_high: Threshold for high volatility classification.
            volatility_threshold_low: Threshold for low volatility classification.
            trend_threshold: Minimum trend strength to classify as trending.
        """
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low
        self.trend_threshold = trend_threshold
        logger.info(
            "Initialized MarketRegimeDetector",
            vol_high=volatility_threshold_high,
            vol_low=volatility_threshold_low,
            trend_threshold=trend_threshold,
        )

    async def detect_regime(
        self,
        symbol: str | None = None,
        lookback_days: int = 20,
    ) -> MarketRegime:
        """Detect current market regime.

        This method analyzes:
        1. Volatility (realized volatility or VIX proxy)
        2. Trend direction and strength
        3. Market breadth indicators
        4. Correlation patterns

        Args:
            symbol: Stock symbol to analyze. If None, analyze market-wide regime.
            lookback_days: Number of days to look back for regime detection.

        Returns:
            Detected market regime.
        """
        # TODO: This is a placeholder implementation
        # In production, this would:
        # 1. Fetch price data from database
        # 2. Calculate volatility metrics
        # 3. Calculate trend indicators
        # 4. Analyze breadth and correlation
        # 5. Classify regime based on all factors

        # For now, return a default regime
        # This will be implemented when integrated with data layer
        logger.info(
            "Detecting market regime",
            symbol=symbol or "MARKET",
            lookback_days=lookback_days,
        )

        # Placeholder: Return sideways as default
        return MarketRegime.SIDEWAYS

    def _calculate_volatility(self, returns: list[float]) -> float:
        """Calculate annualized volatility from returns.

        Args:
            returns: List of periodic returns.

        Returns:
            Annualized volatility.
        """
        if not returns or len(returns) < 2:
            return 0.0

        # Calculate standard deviation
        std_dev = statistics.stdev(returns)

        # Annualize (assuming daily returns)
        # 252 trading days per year
        annualized_vol: float = std_dev * (252**0.5)

        logger.debug(
            "Calculated volatility",
            returns_count=len(returns),
            std_dev=std_dev,
            annualized_vol=annualized_vol,
        )

        return annualized_vol

    def _calculate_trend(self, prices: list[float]) -> tuple[str, float]:
        """Calculate trend direction and strength.

        Args:
            prices: List of prices.

        Returns:
            Tuple of (direction, strength) where:
                - direction is 'up', 'down', or 'sideways'
                - strength is a value between 0 and 1
        """
        if not prices or len(prices) < 2:
            return ("sideways", 0.0)

        # Calculate simple linear regression slope
        n = len(prices)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(prices)

        # Calculate slope
        numerator = sum((x[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return ("sideways", 0.0)

        slope = numerator / denominator

        # Normalize slope to get strength
        price_range = max(prices) - min(prices)
        if price_range == 0:
            return ("sideways", 0.0)

        # Normalize slope by price range and period
        normalized_slope = abs(slope) / (y_mean / n) if y_mean != 0 else 0.0
        strength = min(normalized_slope, 1.0)

        # Determine direction
        if strength < self.trend_threshold:
            direction = "sideways"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"

        logger.debug(
            "Calculated trend",
            slope=slope,
            direction=direction,
            strength=strength,
        )

        return (direction, strength)

    def _classify_regime(
        self,
        volatility: float,
        trend_direction: str,
        trend_strength: float,
    ) -> MarketRegime:
        """Classify market regime from indicators.

        Args:
            volatility: Annualized volatility.
            trend_direction: Trend direction ('up', 'down', 'sideways').
            trend_strength: Trend strength (0 to 1).

        Returns:
            Classified market regime.
        """
        # Crisis: Very high volatility
        if volatility > 0.4:
            logger.info("Classified regime as CRISIS", volatility=volatility)
            return MarketRegime.CRISIS

        # Volatile: High volatility regardless of trend
        if volatility > self.volatility_threshold_high:
            logger.info("Classified regime as VOLATILE", volatility=volatility)
            return MarketRegime.VOLATILE

        # Sideways: Low volatility and weak trend
        if volatility < self.volatility_threshold_low and trend_strength < self.trend_threshold:
            logger.info(
                "Classified regime as SIDEWAYS",
                volatility=volatility,
                trend_strength=trend_strength,
            )
            return MarketRegime.SIDEWAYS

        # Bull: Strong uptrend
        if trend_direction == "up" and trend_strength > self.trend_threshold:
            logger.info(
                "Classified regime as BULL",
                trend_direction=trend_direction,
                trend_strength=trend_strength,
            )
            return MarketRegime.BULL

        # Bear: Strong downtrend
        if trend_direction == "down" and trend_strength > self.trend_threshold:
            logger.info(
                "Classified regime as BEAR",
                trend_direction=trend_direction,
                trend_strength=trend_strength,
            )
            return MarketRegime.BEAR

        # Default to sideways if no clear regime
        logger.info("Classified regime as SIDEWAYS (default)")
        return MarketRegime.SIDEWAYS

    def get_regime_characteristics(self, regime: MarketRegime) -> dict[str, str | float]:
        """Get typical characteristics for a regime.

        Args:
            regime: Market regime.

        Returns:
            Dictionary of regime characteristics.
        """
        characteristics: dict[MarketRegime, dict[str, str | float]] = {
            MarketRegime.BULL: {
                "description": "Strong upward trend with positive momentum",
                "risk_level": "low",
                "position_sizing_multiplier": 1.2,
                "recommended_holding_period": "medium-long",
            },
            MarketRegime.BEAR: {
                "description": "Strong downward trend with negative momentum",
                "risk_level": "high",
                "position_sizing_multiplier": 0.8,
                "recommended_holding_period": "short",
            },
            MarketRegime.SIDEWAYS: {
                "description": "Range-bound market with no clear trend",
                "risk_level": "medium",
                "position_sizing_multiplier": 1.0,
                "recommended_holding_period": "short-medium",
            },
            MarketRegime.VOLATILE: {
                "description": "High volatility with uncertain direction",
                "risk_level": "very_high",
                "position_sizing_multiplier": 0.7,
                "recommended_holding_period": "very_short",
            },
            MarketRegime.CRISIS: {
                "description": "Extreme volatility and risk, potential market dislocation",
                "risk_level": "extreme",
                "position_sizing_multiplier": 0.5,
                "recommended_holding_period": "defensive",
            },
        }

        default_chars: dict[str, str | float] = {
            "description": "Unknown regime",
            "risk_level": "unknown",
            "position_sizing_multiplier": 1.0,
            "recommended_holding_period": "medium",
        }

        return characteristics.get(regime, default_chars)
