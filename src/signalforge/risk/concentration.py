"""Portfolio concentration analysis and alert system.

This module provides comprehensive portfolio concentration monitoring using the
Herfindahl-Hirschman Index (HHI) and multiple concentration metrics. It generates
actionable alerts when portfolio concentration exceeds configurable thresholds.

Key Features:
- HHI calculation for overall portfolio concentration
- Single position concentration limits
- Sector concentration monitoring
- Correlated asset group detection
- Configurable warning and critical thresholds
- Clear, actionable alert messages

Examples:
    Basic concentration analysis:

    >>> from decimal import Decimal
    >>> from signalforge.risk.concentration import (
    ...     ConcentrationAnalyzer,
    ...     ConcentrationConfig,
    ...     Position,
    ... )
    >>>
    >>> config = ConcentrationConfig(max_single_position_pct=0.20)
    >>> analyzer = ConcentrationAnalyzer(config)
    >>> positions = [
    ...     Position(symbol="AAPL", value=Decimal("50000"), weight=0.5),
    ...     Position(symbol="MSFT", value=Decimal("30000"), weight=0.3),
    ...     Position(symbol="JPM", value=Decimal("20000"), weight=0.2),
    ... ]
    >>> result = analyzer.analyze_portfolio(positions, {})
    >>> print(f"Concentrated: {result.is_concentrated}")
    >>> for alert in result.alerts:
    ...     print(f"{alert.severity}: {alert.message}")

    Sector concentration analysis:

    >>> sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    >>> positions = [
    ...     Position(symbol="AAPL", value=Decimal("40000"), weight=0.4, sector="Technology"),
    ...     Position(symbol="MSFT", value=Decimal("40000"), weight=0.4, sector="Technology"),
    ...     Position(symbol="JPM", value=Decimal("20000"), weight=0.2, sector="Financials"),
    ... ]
    >>> result = analyzer.analyze_portfolio(positions, sector_map)
    >>> print(f"Largest sector: {result.largest_sector_pct:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

import polars as pl

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


class AlertType(Enum):
    """Types of concentration alerts.

    Attributes:
        SINGLE_POSITION_EXCEEDED: A single position exceeds the maximum weight threshold.
        SECTOR_CONCENTRATION: A sector's total weight exceeds the maximum sector threshold.
        CORRELATED_GROUP: A group of correlated assets exceeds the maximum group threshold.
        HHI_WARNING: HHI index exceeds the warning threshold.
        HHI_CRITICAL: HHI index exceeds the critical threshold.
    """

    SINGLE_POSITION_EXCEEDED = "single_position_exceeded"
    SECTOR_CONCENTRATION = "sector_concentration"
    CORRELATED_GROUP = "correlated_group"
    HHI_WARNING = "hhi_warning"
    HHI_CRITICAL = "hhi_critical"


class AlertSeverity(Enum):
    """Severity levels for concentration alerts.

    Attributes:
        WARNING: Advisory alert indicating elevated concentration.
        CRITICAL: Critical alert requiring immediate attention.
    """

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ConcentrationConfig:
    """Configuration for portfolio concentration thresholds.

    Attributes:
        max_single_position_pct: Maximum weight for any single position (0.0-1.0).
            Default is 0.15 (15%).
        max_sector_pct: Maximum total weight for any sector (0.0-1.0).
            Default is 0.30 (30%).
        max_correlated_group_pct: Maximum total weight for correlated asset groups (0.0-1.0).
            Default is 0.40 (40%).
        hhi_warning_threshold: HHI threshold for warning alerts (0.0-1.0).
            Default is 0.15. Values above 0.15 indicate moderate concentration.
        hhi_critical_threshold: HHI threshold for critical alerts (0.0-1.0).
            Default is 0.25. Values above 0.25 indicate high concentration.
    """

    max_single_position_pct: float = 0.15
    max_sector_pct: float = 0.30
    max_correlated_group_pct: float = 0.40
    hhi_warning_threshold: float = 0.15
    hhi_critical_threshold: float = 0.25

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.max_single_position_pct <= 1.0:
            raise ValueError(
                f"max_single_position_pct must be between 0.0 and 1.0, "
                f"got {self.max_single_position_pct}"
            )

        if not 0.0 < self.max_sector_pct <= 1.0:
            raise ValueError(
                f"max_sector_pct must be between 0.0 and 1.0, got {self.max_sector_pct}"
            )

        if not 0.0 < self.max_correlated_group_pct <= 1.0:
            raise ValueError(
                f"max_correlated_group_pct must be between 0.0 and 1.0, "
                f"got {self.max_correlated_group_pct}"
            )

        if not 0.0 <= self.hhi_warning_threshold <= 1.0:
            raise ValueError(
                f"hhi_warning_threshold must be between 0.0 and 1.0, "
                f"got {self.hhi_warning_threshold}"
            )

        if not 0.0 <= self.hhi_critical_threshold <= 1.0:
            raise ValueError(
                f"hhi_critical_threshold must be between 0.0 and 1.0, "
                f"got {self.hhi_critical_threshold}"
            )

        if self.hhi_critical_threshold < self.hhi_warning_threshold:
            raise ValueError(
                f"hhi_critical_threshold ({self.hhi_critical_threshold}) must be >= "
                f"hhi_warning_threshold ({self.hhi_warning_threshold})"
            )

        logger.debug(
            "concentration_config_initialized",
            max_single=self.max_single_position_pct,
            max_sector=self.max_sector_pct,
            max_correlated_group=self.max_correlated_group_pct,
            hhi_warning=self.hhi_warning_threshold,
            hhi_critical=self.hhi_critical_threshold,
        )


@dataclass
class ConcentrationAlert:
    """A concentration alert with details.

    Attributes:
        alert_type: The type of concentration alert.
        severity: The severity level (WARNING or CRITICAL).
        message: Human-readable alert message.
        value: The actual concentration value that triggered the alert.
        threshold: The threshold that was exceeded.
        metadata: Additional metadata about the alert.
    """

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    metadata: dict[str, str | float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alert fields."""
        if not self.message or not self.message.strip():
            raise ValueError("Alert message cannot be empty")

        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Alert value must be between 0.0 and 1.0, got {self.value}")

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Alert threshold must be between 0.0 and 1.0, got {self.threshold}")


@dataclass
class ConcentrationResult:
    """Results of portfolio concentration analysis.

    Attributes:
        hhi_index: Herfindahl-Hirschman Index (0.0-1.0). Higher values indicate
            greater concentration. Values above 0.15 suggest moderate concentration,
            above 0.25 suggest high concentration.
        is_concentrated: True if any concentration threshold was exceeded.
        largest_position_pct: Weight of the largest single position.
        largest_sector_pct: Total weight of the largest sector.
        alerts: List of concentration alerts generated.
        metadata: Additional metadata about the analysis.
    """

    hhi_index: float
    is_concentrated: bool
    largest_position_pct: float
    largest_sector_pct: float
    alerts: list[ConcentrationAlert]
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result fields."""
        if not 0.0 <= self.hhi_index <= 1.0:
            raise ValueError(f"HHI index must be between 0.0 and 1.0, got {self.hhi_index}")

        if not 0.0 <= self.largest_position_pct <= 1.0:
            raise ValueError(
                f"largest_position_pct must be between 0.0 and 1.0, "
                f"got {self.largest_position_pct}"
            )

        if not 0.0 <= self.largest_sector_pct <= 1.0:
            raise ValueError(
                f"largest_sector_pct must be between 0.0 and 1.0, "
                f"got {self.largest_sector_pct}"
            )

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[ConcentrationAlert]:
        """Filter alerts by severity level.

        Args:
            severity: The severity level to filter by.

        Returns:
            List of alerts matching the specified severity.

        Examples:
            >>> result = ConcentrationResult(...)
            >>> critical_alerts = result.get_alerts_by_severity(AlertSeverity.CRITICAL)
            >>> print(f"Critical alerts: {len(critical_alerts)}")
        """
        return [alert for alert in self.alerts if alert.severity == severity]

    def get_alerts_by_type(self, alert_type: AlertType) -> list[ConcentrationAlert]:
        """Filter alerts by type.

        Args:
            alert_type: The alert type to filter by.

        Returns:
            List of alerts matching the specified type.

        Examples:
            >>> result = ConcentrationResult(...)
            >>> position_alerts = result.get_alerts_by_type(AlertType.SINGLE_POSITION_EXCEEDED)
            >>> print(f"Position alerts: {len(position_alerts)}")
        """
        return [alert for alert in self.alerts if alert.alert_type == alert_type]


@dataclass
class Position:
    """A portfolio position with concentration metrics.

    Attributes:
        symbol: The asset symbol or identifier.
        value: The monetary value of the position.
        weight: The position weight as a fraction of total portfolio (0.0-1.0).
        sector: Optional sector classification for the position.
    """

    symbol: str
    value: Decimal
    weight: float
    sector: str | None = None

    def __post_init__(self) -> None:
        """Validate position fields."""
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Position symbol cannot be empty")

        if self.value < Decimal("0"):
            raise ValueError(f"Position value cannot be negative, got {self.value}")

        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Position weight must be between 0.0 and 1.0, got {self.weight}")


class ConcentrationAnalyzer:
    """Analyzer for portfolio concentration metrics and alerts.

    This class provides comprehensive concentration analysis including HHI calculation,
    single position monitoring, sector concentration analysis, and correlated group detection.

    Examples:
        >>> from decimal import Decimal
        >>> config = ConcentrationConfig(max_single_position_pct=0.20)
        >>> analyzer = ConcentrationAnalyzer(config)
        >>> positions = [
        ...     Position(symbol="AAPL", value=Decimal("50000"), weight=0.5),
        ...     Position(symbol="MSFT", value=Decimal("30000"), weight=0.3),
        ... ]
        >>> result = analyzer.analyze_portfolio(positions, {})
        >>> print(f"HHI: {result.hhi_index:.3f}")
    """

    def __init__(self, config: ConcentrationConfig | None = None) -> None:
        """Initialize the concentration analyzer.

        Args:
            config: Configuration for concentration thresholds. If None, uses defaults.
        """
        self._config = config or ConcentrationConfig()

        logger.info(
            "concentration_analyzer_initialized",
            max_single_position=self._config.max_single_position_pct,
            max_sector=self._config.max_sector_pct,
            hhi_warning=self._config.hhi_warning_threshold,
            hhi_critical=self._config.hhi_critical_threshold,
        )

    def calculate_hhi(self, weights: list[float]) -> float:
        """Calculate the Herfindahl-Hirschman Index for portfolio concentration.

        The HHI is calculated as the sum of squared weights:
        HHI = sum(weight_i^2) for all positions

        HHI ranges from 0.0 to 1.0:
        - 0.0: Perfectly diversified (infinite positions with equal weights)
        - 1.0: Completely concentrated (single position)
        - < 0.15: Low concentration (well diversified)
        - 0.15-0.25: Moderate concentration
        - > 0.25: High concentration

        Args:
            weights: List of position weights (must sum to approximately 1.0).

        Returns:
            The Herfindahl-Hirschman Index (0.0-1.0).

        Raises:
            ValueError: If weights list is empty or contains invalid values.

        Examples:
            >>> analyzer = ConcentrationAnalyzer()
            >>> weights = [0.5, 0.3, 0.2]  # Three positions
            >>> hhi = analyzer.calculate_hhi(weights)
            >>> print(f"HHI: {hhi:.3f}")
            HHI: 0.380
        """
        if not weights:
            raise ValueError("Weights list cannot be empty")

        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("All weights must be between 0.0 and 1.0")

        total_weight = sum(weights)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            logger.warning(
                "weights_do_not_sum_to_one",
                total_weight=total_weight,
                num_positions=len(weights),
            )

        hhi = sum(w * w for w in weights)

        logger.debug(
            "hhi_calculated",
            hhi=hhi,
            num_positions=len(weights),
            largest_weight=max(weights) if weights else 0.0,
        )

        return hhi

    def analyze_single_position(self, positions: list[Position]) -> list[ConcentrationAlert]:
        """Analyze individual position concentration and generate alerts.

        Checks each position's weight against the maximum single position threshold.
        Generates CRITICAL alerts for positions exceeding the threshold.

        Args:
            positions: List of portfolio positions to analyze.

        Returns:
            List of concentration alerts for positions exceeding limits.

        Examples:
            >>> analyzer = ConcentrationAnalyzer()
            >>> positions = [
            ...     Position(symbol="AAPL", value=Decimal("60000"), weight=0.6),
            ...     Position(symbol="MSFT", value=Decimal("40000"), weight=0.4),
            ... ]
            >>> alerts = analyzer.analyze_single_position(positions)
            >>> for alert in alerts:
            ...     print(alert.message)
        """
        if not positions:
            logger.debug("no_positions_to_analyze")
            return []

        alerts: list[ConcentrationAlert] = []

        for position in positions:
            if position.weight > self._config.max_single_position_pct:
                alert = ConcentrationAlert(
                    alert_type=AlertType.SINGLE_POSITION_EXCEEDED,
                    severity=AlertSeverity.CRITICAL,
                    message=(
                        f"Position {position.symbol} weight {position.weight:.1%} exceeds "
                        f"maximum single position limit of "
                        f"{self._config.max_single_position_pct:.1%}"
                    ),
                    value=position.weight,
                    threshold=self._config.max_single_position_pct,
                    metadata={
                        "symbol": position.symbol,
                        "value": float(position.value),
                        "sector": position.sector or "Unknown",
                    },
                )
                alerts.append(alert)

                logger.warning(
                    "single_position_limit_exceeded",
                    symbol=position.symbol,
                    weight=position.weight,
                    threshold=self._config.max_single_position_pct,
                )

        logger.debug(
            "single_position_analysis_complete",
            num_positions=len(positions),
            num_alerts=len(alerts),
        )

        return alerts

    def analyze_sector_concentration(
        self,
        positions: list[Position],
        sector_map: dict[str, str],
    ) -> list[ConcentrationAlert]:
        """Analyze sector concentration and generate alerts.

        Aggregates position weights by sector and checks against maximum sector threshold.
        Generates CRITICAL alerts for sectors exceeding the threshold.

        Args:
            positions: List of portfolio positions to analyze.
            sector_map: Mapping from symbol to sector classification.

        Returns:
            List of concentration alerts for sectors exceeding limits.

        Examples:
            >>> analyzer = ConcentrationAnalyzer()
            >>> positions = [
            ...     Position(symbol="AAPL", value=Decimal("40000"), weight=0.4),
            ...     Position(symbol="MSFT", value=Decimal("40000"), weight=0.4),
            ... ]
            >>> sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
            >>> alerts = analyzer.analyze_sector_concentration(positions, sector_map)
        """
        if not positions:
            logger.debug("no_positions_for_sector_analysis")
            return []

        # Aggregate weights by sector
        sector_weights: dict[str, float] = {}
        sector_symbols: dict[str, list[str]] = {}

        for position in positions:
            sector = sector_map.get(position.symbol) if sector_map else None
            if not sector:
                sector = position.sector

            if sector:
                sector_weights[sector] = sector_weights.get(sector, 0.0) + position.weight
                if sector not in sector_symbols:
                    sector_symbols[sector] = []
                sector_symbols[sector].append(position.symbol)

        if not sector_weights:
            logger.debug("no_sector_information_available_skipping_sector_analysis")
            return []

        logger.debug(
            "sector_aggregation_complete",
            num_sectors=len(sector_weights),
            sectors=list(sector_weights.keys()),
        )

        # Generate alerts for sectors exceeding threshold
        alerts: list[ConcentrationAlert] = []

        for sector, weight in sector_weights.items():
            if weight > self._config.max_sector_pct:
                symbols = sector_symbols.get(sector, [])
                alert = ConcentrationAlert(
                    alert_type=AlertType.SECTOR_CONCENTRATION,
                    severity=AlertSeverity.CRITICAL,
                    message=(
                        f"Sector {sector} weight {weight:.1%} exceeds maximum sector limit "
                        f"of {self._config.max_sector_pct:.1%} "
                        f"({len(symbols)} positions: {', '.join(symbols[:3])}"
                        f"{'...' if len(symbols) > 3 else ''})"
                    ),
                    value=weight,
                    threshold=self._config.max_sector_pct,
                    metadata={
                        "sector": sector,
                        "num_positions": len(symbols),
                        "symbols": ", ".join(symbols),
                    },
                )
                alerts.append(alert)

                logger.warning(
                    "sector_concentration_exceeded",
                    sector=sector,
                    weight=weight,
                    threshold=self._config.max_sector_pct,
                    num_positions=len(symbols),
                )

        logger.debug(
            "sector_concentration_analysis_complete",
            num_sectors=len(sector_weights),
            num_alerts=len(alerts),
        )

        return alerts

    def analyze_correlated_groups(
        self,
        positions: list[Position],
        correlation_matrix: pl.DataFrame,
        threshold: float = 0.7,
    ) -> list[ConcentrationAlert]:
        """Analyze correlated asset group concentration and generate alerts.

        Identifies groups of highly correlated assets and checks if their combined
        weight exceeds the maximum correlated group threshold.

        Args:
            positions: List of portfolio positions to analyze.
            correlation_matrix: Polars DataFrame with correlation matrix.
                Must have columns matching position symbols.
            threshold: Correlation threshold for grouping (default 0.7).
                Assets with correlation above this are considered highly correlated.

        Returns:
            List of concentration alerts for correlated groups exceeding limits.

        Raises:
            ValueError: If correlation_matrix format is invalid or threshold is out of range.

        Examples:
            >>> import polars as pl
            >>> analyzer = ConcentrationAnalyzer()
            >>> positions = [...]
            >>> correlation_matrix = pl.DataFrame({
            ...     "symbol": ["AAPL", "MSFT", "JPM"],
            ...     "AAPL": [1.0, 0.85, 0.3],
            ...     "MSFT": [0.85, 1.0, 0.25],
            ...     "JPM": [0.3, 0.25, 1.0],
            ... })
            >>> alerts = analyzer.analyze_correlated_groups(positions, correlation_matrix)
        """
        if not positions:
            logger.debug("no_positions_for_correlation_analysis")
            return []

        if correlation_matrix.is_empty():
            logger.debug("empty_correlation_matrix_skipping_analysis")
            return []

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Correlation threshold must be between 0.0 and 1.0, got {threshold}")

        # Create symbol to position mapping
        symbol_to_position = {pos.symbol: pos for pos in positions}

        # Identify correlated groups using graph-based clustering
        symbols = [pos.symbol for pos in positions]
        visited = set()
        correlated_groups: list[list[str]] = []

        def find_correlated_group(start_symbol: str) -> list[str]:
            """Find all symbols correlated with start_symbol using BFS."""
            group = [start_symbol]
            queue = [start_symbol]
            visited.add(start_symbol)

            while queue:
                current = queue.pop(0)

                # Get correlations for current symbol
                if "symbol" not in correlation_matrix.columns:
                    # Assume index is symbol
                    continue

                for other_symbol in symbols:
                    if other_symbol in visited or other_symbol == current:
                        continue

                    # Get correlation value
                    try:
                        corr_row = correlation_matrix.filter(
                            pl.col("symbol") == current
                        ).select(pl.col(other_symbol))

                        if not corr_row.is_empty():
                            corr_value = corr_row.to_numpy()[0, 0]

                            if abs(corr_value) >= threshold:
                                group.append(other_symbol)
                                queue.append(other_symbol)
                                visited.add(other_symbol)
                    except Exception as e:
                        logger.warning(
                            "correlation_lookup_failed",
                            current=current,
                            other=other_symbol,
                            error=str(e),
                        )
                        continue

            return group

        # Find all correlated groups
        for symbol in symbols:
            if symbol not in visited:
                group = find_correlated_group(symbol)
                if len(group) > 1:  # Only consider groups with multiple assets
                    correlated_groups.append(group)

        logger.debug(
            "correlated_groups_identified",
            num_groups=len(correlated_groups),
            groups=[len(g) for g in correlated_groups],
        )

        # Generate alerts for groups exceeding threshold
        alerts: list[ConcentrationAlert] = []

        for group in correlated_groups:
            group_weight = sum(
                symbol_to_position[symbol].weight
                for symbol in group
                if symbol in symbol_to_position
            )

            if group_weight > self._config.max_correlated_group_pct:
                alert = ConcentrationAlert(
                    alert_type=AlertType.CORRELATED_GROUP,
                    severity=AlertSeverity.CRITICAL,
                    message=(
                        f"Correlated group ({', '.join(group[:3])}"
                        f"{'...' if len(group) > 3 else ''}) "
                        f"weight {group_weight:.1%} exceeds maximum correlated group limit "
                        f"of {self._config.max_correlated_group_pct:.1%}"
                    ),
                    value=group_weight,
                    threshold=self._config.max_correlated_group_pct,
                    metadata={
                        "group_size": len(group),
                        "symbols": ", ".join(group),
                        "correlation_threshold": threshold,
                    },
                )
                alerts.append(alert)

                logger.warning(
                    "correlated_group_concentration_exceeded",
                    group_size=len(group),
                    weight=group_weight,
                    threshold=self._config.max_correlated_group_pct,
                )

        logger.debug(
            "correlation_analysis_complete",
            num_groups=len(correlated_groups),
            num_alerts=len(alerts),
        )

        return alerts

    def analyze_portfolio(
        self,
        positions: list[Position],
        sector_map: dict[str, str],
        correlation_matrix: pl.DataFrame | None = None,
    ) -> ConcentrationResult:
        """Perform comprehensive portfolio concentration analysis.

        This method combines all concentration analyses:
        - HHI calculation
        - Single position concentration
        - Sector concentration
        - Correlated group concentration (if correlation matrix provided)

        Args:
            positions: List of portfolio positions to analyze.
            sector_map: Mapping from symbol to sector classification.
            correlation_matrix: Optional Polars DataFrame with asset correlations.

        Returns:
            ConcentrationResult containing HHI, alerts, and analysis metadata.

        Raises:
            ValueError: If positions list is empty or contains invalid data.

        Examples:
            >>> from decimal import Decimal
            >>> analyzer = ConcentrationAnalyzer()
            >>> positions = [
            ...     Position(symbol="AAPL", value=Decimal("50000"), weight=0.5),
            ...     Position(symbol="MSFT", value=Decimal("30000"), weight=0.3),
            ...     Position(symbol="JPM", value=Decimal("20000"), weight=0.2),
            ... ]
            >>> sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
            >>> result = analyzer.analyze_portfolio(positions, sector_map)
            >>> print(f"HHI: {result.hhi_index:.3f}, Alerts: {len(result.alerts)}")
        """
        if not positions:
            raise ValueError("Cannot analyze empty portfolio")

        logger.info(
            "starting_portfolio_concentration_analysis",
            num_positions=len(positions),
            has_sector_map=bool(sector_map),
            has_correlation_matrix=correlation_matrix is not None,
        )

        # Calculate HHI
        weights = [pos.weight for pos in positions]
        hhi = self.calculate_hhi(weights)

        # Collect all alerts
        all_alerts: list[ConcentrationAlert] = []

        # Analyze single position concentration
        position_alerts = self.analyze_single_position(positions)
        all_alerts.extend(position_alerts)

        # Analyze sector concentration
        sector_alerts = self.analyze_sector_concentration(positions, sector_map)
        all_alerts.extend(sector_alerts)

        # Analyze correlated groups if correlation matrix provided
        if correlation_matrix is not None:
            correlation_alerts = self.analyze_correlated_groups(positions, correlation_matrix)
            all_alerts.extend(correlation_alerts)

        # Generate HHI alerts
        if hhi >= self._config.hhi_critical_threshold:
            all_alerts.append(
                ConcentrationAlert(
                    alert_type=AlertType.HHI_CRITICAL,
                    severity=AlertSeverity.CRITICAL,
                    message=(
                        f"Portfolio HHI {hhi:.3f} exceeds critical threshold "
                        f"of {self._config.hhi_critical_threshold:.3f}. "
                        f"Portfolio is highly concentrated."
                    ),
                    value=hhi,
                    threshold=self._config.hhi_critical_threshold,
                )
            )
            logger.warning("hhi_critical_threshold_exceeded", hhi=hhi)
        elif hhi >= self._config.hhi_warning_threshold:
            all_alerts.append(
                ConcentrationAlert(
                    alert_type=AlertType.HHI_WARNING,
                    severity=AlertSeverity.WARNING,
                    message=(
                        f"Portfolio HHI {hhi:.3f} exceeds warning threshold "
                        f"of {self._config.hhi_warning_threshold:.3f}. "
                        f"Consider diversifying holdings."
                    ),
                    value=hhi,
                    threshold=self._config.hhi_warning_threshold,
                )
            )
            logger.info("hhi_warning_threshold_exceeded", hhi=hhi)

        # Calculate largest position and sector weights
        largest_position_pct = max(weights) if weights else 0.0

        sector_weights: dict[str, float] = {}
        for position in positions:
            sector = sector_map.get(position.symbol) or position.sector
            if sector:
                sector_weights[sector] = sector_weights.get(sector, 0.0) + position.weight

        largest_sector_pct = max(sector_weights.values()) if sector_weights else 0.0

        # Determine if portfolio is concentrated
        is_concentrated = len(all_alerts) > 0

        result = ConcentrationResult(
            hhi_index=hhi,
            is_concentrated=is_concentrated,
            largest_position_pct=largest_position_pct,
            largest_sector_pct=largest_sector_pct,
            alerts=all_alerts,
            metadata={
                "num_positions": len(positions),
                "num_sectors": len(sector_weights),
                "total_value": float(sum(pos.value for pos in positions)),
                "num_warnings": len([a for a in all_alerts if a.severity == AlertSeverity.WARNING]),
                "num_critical": len(
                    [a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]
                ),
            },
        )

        logger.info(
            "portfolio_concentration_analysis_complete",
            hhi=hhi,
            is_concentrated=is_concentrated,
            num_alerts=len(all_alerts),
            num_warnings=result.metadata["num_warnings"],
            num_critical=result.metadata["num_critical"],
        )

        return result
