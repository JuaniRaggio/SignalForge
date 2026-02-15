"""Value at Risk (VaR) calculation module for portfolio risk assessment.

This module provides comprehensive VaR calculation functionality including:
- Historical VaR: Based on historical return percentiles
- Parametric VaR: Assumes normal distribution of returns
- Monte Carlo VaR: Simulation-based approach
- Expected Shortfall (CVaR): Average loss beyond VaR threshold
- Portfolio VaR: Aggregated VaR considering correlations

All methods follow standard financial risk management practices.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

import polars as pl
from scipy import stats

from signalforge.core.exceptions import ValidationError
from signalforge.core.logging import LoggerMixin


class VaRMethod(str, Enum):
    """Supported VaR calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True)
class VaRConfig:
    """Configuration for VaR calculations.

    Attributes:
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR).
        horizon_days: Time horizon in days for VaR calculation.
        method: VaR calculation method to use.
    """

    confidence_level: float = 0.95
    horizon_days: int = 1
    method: VaRMethod = VaRMethod.HISTORICAL

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValidationError(
                "Confidence level must be between 0 and 1",
                field="confidence_level",
                value=self.confidence_level,
                constraint="0 < confidence_level < 1",
            )
        if self.horizon_days < 1:
            raise ValidationError(
                "Horizon days must be at least 1",
                field="horizon_days",
                value=self.horizon_days,
                constraint="horizon_days >= 1",
            )


@dataclass(frozen=True)
class VaRResult:
    """Result of a VaR calculation.

    Attributes:
        var_amount: Value at Risk in absolute terms.
        var_pct: Value at Risk as a percentage.
        confidence_level: Confidence level used for calculation.
        horizon_days: Time horizon used for calculation.
        method: Method used for calculation.
        expected_shortfall: Expected Shortfall (CVaR) - average loss beyond VaR.
    """

    var_amount: Decimal
    var_pct: float
    confidence_level: float
    horizon_days: int
    method: VaRMethod
    expected_shortfall: Decimal


@dataclass(frozen=True)
class PositionVaR:
    """VaR information for a single position in a portfolio.

    Attributes:
        symbol: Asset symbol/ticker.
        weight: Position weight in portfolio (sum of all weights should be 1.0).
        var_individual: Individual VaR for this position.
    """

    symbol: str
    weight: float
    var_individual: Decimal

    def __post_init__(self) -> None:
        """Validate position parameters."""
        if not 0 <= self.weight <= 1:
            raise ValidationError(
                "Position weight must be between 0 and 1",
                field="weight",
                value=self.weight,
                constraint="0 <= weight <= 1",
            )


class VaRCalculator(LoggerMixin):
    """Value at Risk calculator with multiple calculation methods.

    This class provides comprehensive VaR calculation functionality for both
    individual assets and portfolios, with support for multiple calculation
    methodologies and proper edge case handling.
    """

    def __init__(self, config: VaRConfig) -> None:
        """Initialize VaR calculator.

        Args:
            config: VaR calculation configuration.
        """
        self.config = config
        self.logger.info(
            "var_calculator_initialized",
            confidence_level=config.confidence_level,
            horizon_days=config.horizon_days,
            method=config.method.value,
        )

    def calculate_returns(self, prices: pl.Series) -> pl.Series:
        """Calculate log returns from price series.

        Uses log returns for better statistical properties and additivity
        over time periods.

        Args:
            prices: Price series (must be positive values).

        Returns:
            Series of log returns.

        Raises:
            ValidationError: If prices contain non-positive values.
        """
        if len(prices) == 0:
            self.logger.warning("calculate_returns_empty_series")
            return pl.Series("returns", [], dtype=pl.Float64)

        if prices.min() <= 0:  # type: ignore[operator]
            raise ValidationError(
                "Prices must be positive for return calculation",
                field="prices",
                constraint="all prices > 0",
            )

        # Calculate log returns: ln(P_t / P_{t-1})
        returns = (prices / prices.shift(1)).log()

        # Drop the first NaN value
        returns = returns.slice(1)

        # Log return statistics
        if len(returns) > 0:
            mean_val = returns.mean()
            std_val = returns.std()
            self.logger.debug(
                "returns_calculated",
                num_returns=len(returns),
                mean_return=float(mean_val) if mean_val is not None else None,  # type: ignore[arg-type]
                std_return=float(std_val) if std_val is not None else None,  # type: ignore[arg-type]
            )

        return returns

    def calculate_historical_var(
        self, returns: pl.Series, portfolio_value: Decimal
    ) -> VaRResult:
        """Calculate VaR using historical simulation method.

        This method uses the empirical distribution of returns to determine
        VaR without making distributional assumptions.

        Args:
            returns: Historical returns series.
            portfolio_value: Current portfolio value.

        Returns:
            VaR calculation result.

        Raises:
            ValidationError: If insufficient data is available.
        """
        if len(returns) < 2:
            raise ValidationError(
                "Insufficient data for historical VaR calculation",
                field="returns",
                value=len(returns),
                constraint="at least 2 returns required",
            )

        # Calculate the percentile corresponding to the loss threshold
        # For 95% confidence, we want the 5th percentile (worst 5%)
        alpha = 1 - self.config.confidence_level

        # Scale returns by horizon (assumes i.i.d. returns)
        horizon_scaling = (self.config.horizon_days**0.5) if self.config.horizon_days > 1 else 1.0
        scaled_returns = returns * horizon_scaling

        # VaR is the percentile of the loss distribution
        var_percentile = float(scaled_returns.quantile(alpha))  # type: ignore[arg-type]

        # Calculate Expected Shortfall (average of losses beyond VaR)
        es = self._calculate_expected_shortfall(scaled_returns, var_percentile)

        # Convert to monetary terms (VaR is always reported as a positive number)
        var_amount = Decimal(str(abs(var_percentile))) * portfolio_value
        es_amount = Decimal(str(abs(es))) * portfolio_value

        self.logger.info(
            "historical_var_calculated",
            var_amount=float(var_amount),
            var_pct=abs(var_percentile),
            expected_shortfall=float(es_amount),
            num_returns=len(returns),
        )

        return VaRResult(
            var_amount=var_amount,
            var_pct=abs(var_percentile),
            confidence_level=self.config.confidence_level,
            horizon_days=self.config.horizon_days,
            method=VaRMethod.HISTORICAL,
            expected_shortfall=es_amount,
        )

    def calculate_parametric_var(
        self, returns: pl.Series, portfolio_value: Decimal
    ) -> VaRResult:
        """Calculate VaR using parametric (variance-covariance) method.

        Assumes returns follow a normal distribution and uses analytical
        formulas for VaR calculation.

        Args:
            returns: Historical returns series.
            portfolio_value: Current portfolio value.

        Returns:
            VaR calculation result.

        Raises:
            ValidationError: If insufficient data is available.
        """
        if len(returns) < 2:
            raise ValidationError(
                "Insufficient data for parametric VaR calculation",
                field="returns",
                value=len(returns),
                constraint="at least 2 returns required",
            )

        # Calculate return statistics
        mu = float(returns.mean())  # type: ignore[arg-type]
        sigma = float(returns.std())  # type: ignore[arg-type]

        # Handle edge case of constant returns (zero volatility)
        if sigma == 0 or sigma is None:
            self.logger.warning(
                "zero_volatility_detected",
                mean_return=mu,
                message="Returns have zero volatility, VaR will be based on mean only",
            )
            sigma = 1e-10  # Small epsilon to avoid division by zero

        # Get z-score for confidence level (e.g., 1.645 for 95% one-tailed)
        alpha = 1 - self.config.confidence_level
        z_score = stats.norm.ppf(alpha)

        # VaR formula: mu - z * sigma * sqrt(horizon)
        # Note: z_score is negative for left tail, so we subtract it
        horizon_scaling = self.config.horizon_days**0.5
        var_pct = mu - z_score * sigma * horizon_scaling

        # Ensure VaR is non-negative
        var_pct = max(var_pct, 0.0)

        var_amount = Decimal(str(var_pct)) * portfolio_value

        # For parametric VaR, Expected Shortfall has a closed form
        # ES = sigma * sqrt(h) * phi(z) / alpha - mu
        # where phi is the standard normal PDF
        # This gives the expected loss beyond the VaR threshold
        pdf_at_z = stats.norm.pdf(z_score)
        es_pct = sigma * horizon_scaling * (pdf_at_z / alpha) - mu
        es_pct = abs(es_pct)  # ES is always reported as positive
        # Ensure ES is at least as large as VaR
        es_pct = max(es_pct, var_pct)
        es_amount = Decimal(str(es_pct)) * portfolio_value

        self.logger.info(
            "parametric_var_calculated",
            var_amount=float(var_amount),
            var_pct=var_pct,
            expected_shortfall=float(es_amount),
            mu=mu,
            sigma=sigma,
            z_score=z_score,
        )

        return VaRResult(
            var_amount=var_amount,
            var_pct=var_pct,
            confidence_level=self.config.confidence_level,
            horizon_days=self.config.horizon_days,
            method=VaRMethod.PARAMETRIC,
            expected_shortfall=es_amount,
        )

    def calculate_monte_carlo_var(
        self, returns: pl.Series, portfolio_value: Decimal, simulations: int = 10000
    ) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation.

        Simulates future returns based on historical statistics and uses
        the empirical distribution of simulated portfolio values.

        Args:
            returns: Historical returns series.
            portfolio_value: Current portfolio value.
            simulations: Number of Monte Carlo simulations to run.

        Returns:
            VaR calculation result.

        Raises:
            ValidationError: If insufficient data or invalid simulation count.
        """
        if len(returns) < 2:
            raise ValidationError(
                "Insufficient data for Monte Carlo VaR calculation",
                field="returns",
                value=len(returns),
                constraint="at least 2 returns required",
            )

        if simulations < 100:
            raise ValidationError(
                "Insufficient simulations for Monte Carlo VaR",
                field="simulations",
                value=simulations,
                constraint="at least 100 simulations required",
            )

        # Estimate distribution parameters from historical data
        mu = float(returns.mean())  # type: ignore[arg-type]
        sigma = float(returns.std())  # type: ignore[arg-type]

        # Handle edge case of constant returns
        if sigma == 0 or sigma is None:
            self.logger.warning(
                "zero_volatility_detected_mc",
                mean_return=mu,
                message="Returns have zero volatility in Monte Carlo simulation",
            )
            sigma = 1e-10

        # Simulate returns for the horizon period
        # For multi-day horizons, sum daily returns (log returns are additive)
        horizon_scaling = self.config.horizon_days**0.5
        simulated_returns = stats.norm.rvs(
            loc=mu * self.config.horizon_days,
            scale=sigma * horizon_scaling,
            size=simulations,
            random_state=42,  # For reproducibility
        )

        # Calculate simulated portfolio values
        simulated_series = pl.Series("simulated_returns", simulated_returns)

        # VaR is the percentile of the simulated loss distribution
        alpha = 1 - self.config.confidence_level
        var_percentile = float(simulated_series.quantile(alpha))  # type: ignore[arg-type]

        # Calculate Expected Shortfall from simulations
        es = self._calculate_expected_shortfall(simulated_series, var_percentile)

        # Convert to monetary terms
        var_amount = Decimal(str(abs(var_percentile))) * portfolio_value
        es_amount = Decimal(str(abs(es))) * portfolio_value

        self.logger.info(
            "monte_carlo_var_calculated",
            var_amount=float(var_amount),
            var_pct=abs(var_percentile),
            expected_shortfall=float(es_amount),
            simulations=simulations,
            mu=mu,
            sigma=sigma,
        )

        return VaRResult(
            var_amount=var_amount,
            var_pct=abs(var_percentile),
            confidence_level=self.config.confidence_level,
            horizon_days=self.config.horizon_days,
            method=VaRMethod.MONTE_CARLO,
            expected_shortfall=es_amount,
        )

    def _calculate_expected_shortfall(self, returns: pl.Series, var_percentile: float) -> float:
        """Calculate Expected Shortfall (CVaR) from returns.

        Expected Shortfall is the average of all losses that exceed the VaR threshold.

        Args:
            returns: Returns series.
            var_percentile: VaR percentile threshold.

        Returns:
            Expected Shortfall value.
        """
        # Filter returns worse than VaR threshold
        tail_losses = returns.filter(returns <= var_percentile)

        if len(tail_losses) == 0:
            # If no losses exceed VaR, ES equals VaR
            return var_percentile

        # Average of tail losses
        es = float(tail_losses.mean())  # type: ignore[arg-type]
        return es

    def calculate_portfolio_var(
        self, positions: list[PositionVaR], correlations: pl.DataFrame
    ) -> VaRResult:
        """Calculate portfolio VaR considering position correlations.

        Uses the variance-covariance method to aggregate individual position
        VaRs into a portfolio VaR, accounting for diversification effects.

        Args:
            positions: List of position VaR information.
            correlations: Correlation matrix between positions (symbol x symbol).

        Returns:
            Portfolio VaR result.

        Raises:
            ValidationError: If inputs are invalid or inconsistent.
        """
        if len(positions) == 0:
            raise ValidationError(
                "Cannot calculate portfolio VaR with no positions",
                field="positions",
                constraint="at least 1 position required",
            )

        # Validate weights sum to approximately 1.0
        total_weight = sum(pos.weight for pos in positions)
        if not (0.99 <= total_weight <= 1.01):
            raise ValidationError(
                "Position weights must sum to 1.0",
                field="weights",
                value=total_weight,
                constraint="sum of weights should be 1.0",
            )

        # For a single position, return its VaR directly
        if len(positions) == 1:
            pos = positions[0]
            self.logger.info(
                "single_position_portfolio_var",
                symbol=pos.symbol,
                var_amount=float(pos.var_individual),
            )
            return VaRResult(
                var_amount=pos.var_individual,
                var_pct=0.0,  # Not applicable for portfolio VaR
                confidence_level=self.config.confidence_level,
                horizon_days=self.config.horizon_days,
                method=self.config.method,
                expected_shortfall=pos.var_individual,  # Conservative estimate
            )

        # Validate correlation matrix
        if correlations.shape[0] != len(positions) or correlations.shape[1] != len(positions):
            raise ValidationError(
                "Correlation matrix dimensions must match number of positions",
                field="correlations",
                value=f"{correlations.shape[0]}x{correlations.shape[1]}",
                constraint=f"{len(positions)}x{len(positions)}",
            )

        # Build weight and VaR vectors
        weights = pl.Series("weights", [pos.weight for pos in positions])
        var_values = pl.Series("vars", [float(pos.var_individual) for pos in positions])

        # Calculate portfolio variance using matrix operations
        # Portfolio Variance = w^T * Cov * w
        # where Cov = diag(vars) * Corr * diag(vars)

        # Convert to numpy for matrix operations
        import numpy as np

        w = weights.to_numpy()
        v = var_values.to_numpy()
        corr = correlations.to_numpy()

        # Create covariance matrix from VaRs and correlations
        # This is an approximation: Cov[i,j] = VaR[i] * VaR[j] * Corr[i,j]
        var_matrix = np.outer(v, v)
        cov_matrix = var_matrix * corr

        # Calculate portfolio VaR using matrix multiplication
        portfolio_variance = w.T @ cov_matrix @ w
        portfolio_var = np.sqrt(portfolio_variance)

        portfolio_var_amount = Decimal(str(float(portfolio_var)))

        # Conservative estimate for portfolio ES (use max individual ES scaled by diversification)
        diversification_factor = portfolio_var / sum(w[i] * v[i] for i in range(len(w)))
        max_individual_var = max(pos.var_individual for pos in positions)
        portfolio_es = max_individual_var * Decimal(str(float(diversification_factor)))

        self.logger.info(
            "portfolio_var_calculated",
            num_positions=len(positions),
            portfolio_var=float(portfolio_var_amount),
            diversification_benefit=float(sum(w[i] * v[i] for i in range(len(w))))
            - float(portfolio_var),
        )

        return VaRResult(
            var_amount=portfolio_var_amount,
            var_pct=0.0,  # Not applicable for portfolio VaR
            confidence_level=self.config.confidence_level,
            horizon_days=self.config.horizon_days,
            method=self.config.method,
            expected_shortfall=portfolio_es,
        )

    def calculate(self, returns: pl.Series, portfolio_value: Decimal) -> VaRResult:
        """Calculate VaR using the configured method.

        Convenience method that delegates to the appropriate calculation
        method based on configuration.

        Args:
            returns: Historical returns series.
            portfolio_value: Current portfolio value.

        Returns:
            VaR calculation result.

        Raises:
            ValidationError: If method is unsupported or inputs are invalid.
        """
        if self.config.method == VaRMethod.HISTORICAL:
            return self.calculate_historical_var(returns, portfolio_value)
        elif self.config.method == VaRMethod.PARAMETRIC:
            return self.calculate_parametric_var(returns, portfolio_value)
        elif self.config.method == VaRMethod.MONTE_CARLO:
            return self.calculate_monte_carlo_var(returns, portfolio_value)
        else:
            raise ValidationError(
                f"Unsupported VaR method: {self.config.method}",
                field="method",
                value=self.config.method,
            )
