"""Correlation matrix calculation and analysis for portfolio risk management.

This module implements correlation matrix calculation, analysis, and breakdown detection
for multi-asset portfolios. It provides tools for identifying high correlations,
detecting correlation breakdowns between time periods, and analyzing portfolio
diversification through correlation clustering.

The implementation uses Polars for efficient data processing and supports both
Pearson and Spearman correlation coefficients. It can detect regime-dependent
correlation changes and provide alerts when correlation structures shift significantly.

Examples:
    Basic correlation matrix calculation:

    >>> import polars as pl
    >>> from signalforge.risk.correlation import CorrelationCalculator, CorrelationConfig
    >>>
    >>> config = CorrelationConfig(lookback_days=60)
    >>> calculator = CorrelationCalculator(config)
    >>>
    >>> prices = pl.DataFrame({
    ...     "timestamp": pl.date_range(start="2023-01-01", periods=100, interval="1d"),
    ...     "AAPL": [100.0 + i * 0.5 for i in range(100)],
    ...     "MSFT": [200.0 + i * 0.3 for i in range(100)],
    ...     "GOOGL": [150.0 + i * 0.4 for i in range(100)],
    ... })
    >>>
    >>> returns = calculator.calculate_returns(prices)
    >>> matrix = calculator.calculate_correlation_matrix(returns)
    >>> high_corr = calculator.find_high_correlations(matrix, threshold=0.7)

    Detecting correlation breakdown:

    >>> previous_matrix = calculator.calculate_correlation_matrix(returns.head(50))
    >>> current_matrix = calculator.calculate_correlation_matrix(returns.tail(50))
    >>> breakdowns = calculator.detect_correlation_breakdown(
    ...     current_matrix, previous_matrix, threshold=0.3
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation calculation.

    Attributes:
        lookback_days: Number of historical days to use for correlation calculation.
            Default is 60.
        min_observations: Minimum number of observations required for valid correlation.
            Default is 20.
        alert_threshold: Threshold for high correlation alerts. Correlations with
            absolute value above this threshold will trigger alerts. Default is 0.7.
    """

    lookback_days: int = 60
    min_observations: int = 20
    alert_threshold: float = 0.7

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.lookback_days < 2:
            raise ValueError(f"lookback_days must be at least 2, got {self.lookback_days}")
        if self.min_observations < 2:
            raise ValueError(f"min_observations must be at least 2, got {self.min_observations}")
        if self.min_observations > self.lookback_days:
            raise ValueError(
                f"min_observations ({self.min_observations}) cannot exceed "
                f"lookback_days ({self.lookback_days})"
            )
        if not 0.0 <= self.alert_threshold <= 1.0:
            raise ValueError(
                f"alert_threshold must be between 0.0 and 1.0, got {self.alert_threshold}"
            )


@dataclass
class CorrelationResult:
    """Result of correlation matrix calculation.

    Attributes:
        matrix: Correlation matrix as a Polars DataFrame with symbol names as columns.
        high_correlations: List of tuples (symbol1, symbol2, correlation) where
            absolute correlation exceeds the alert threshold.
        avg_correlation: Average absolute correlation across all symbol pairs.
        timestamp: Timestamp when the correlation was calculated.
    """

    matrix: pl.DataFrame
    high_correlations: list[tuple[str, str, float]]
    avg_correlation: float
    timestamp: datetime


@dataclass
class CorrelationBreakdown:
    """Information about a correlation breakdown between periods.

    Attributes:
        symbol_pair: Tuple of two symbol names (symbol1, symbol2).
        previous_corr: Correlation coefficient in the previous period.
        current_corr: Correlation coefficient in the current period.
        change: Absolute change in correlation (current - previous).
        regime: Optional market regime when breakdown occurred.
    """

    symbol_pair: tuple[str, str]
    previous_corr: float
    current_corr: float
    change: float
    regime: str | None = None


def pearson_correlation(x: pl.Series, y: pl.Series) -> float:
    """Calculate Pearson correlation coefficient between two series.

    The Pearson correlation measures the linear relationship between two variables.
    It ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation),
    with 0 indicating no linear relationship.

    Args:
        x: First data series.
        y: Second data series.

    Returns:
        Pearson correlation coefficient.

    Raises:
        ValueError: If series have different lengths or contain all NaN values.
    """
    if x.len() != y.len():
        raise ValueError(f"Series must have same length: {x.len()} != {y.len()}")

    # Drop pairs where either value is null
    df = pl.DataFrame({"x": x, "y": y}).drop_nulls()

    if df.height == 0:
        raise ValueError("No valid observations after dropping nulls")

    x_clean = df["x"]
    y_clean = df["y"]

    # Check for constant series
    if x_clean.std() == 0.0 or y_clean.std() == 0.0:
        return 0.0

    # Calculate Pearson correlation using Polars
    corr_df = pl.DataFrame({"x": x_clean, "y": y_clean}).select(
        pl.corr("x", "y").alias("correlation")
    )
    correlation = float(corr_df["correlation"][0])

    return correlation


def spearman_correlation(x: pl.Series, y: pl.Series) -> float:
    """Calculate Spearman rank correlation coefficient between two series.

    The Spearman correlation measures the monotonic relationship between two variables
    by computing the Pearson correlation of the rank values. It is more robust to
    outliers than Pearson correlation.

    Args:
        x: First data series.
        y: Second data series.

    Returns:
        Spearman correlation coefficient.

    Raises:
        ValueError: If series have different lengths or contain all NaN values.
    """
    if x.len() != y.len():
        raise ValueError(f"Series must have same length: {x.len()} != {y.len()}")

    # Drop pairs where either value is null
    df = pl.DataFrame({"x": x, "y": y}).drop_nulls()

    if df.height == 0:
        raise ValueError("No valid observations after dropping nulls")

    x_clean = df["x"].to_numpy()
    y_clean = df["y"].to_numpy()

    # Use scipy's spearmanr
    corr, _ = spearmanr(x_clean, y_clean)

    # Handle NaN result (can occur with constant series)
    if np.isnan(corr):
        return 0.0

    return float(corr)


class CorrelationCalculator:
    """Calculator for correlation matrices and correlation-based risk metrics.

    This class provides methods for calculating correlation matrices, detecting
    high correlations, identifying correlation breakdowns, and analyzing portfolio
    diversification through correlation clustering.

    Attributes:
        config: Configuration parameters for correlation calculation.

    Examples:
        Basic usage:

        >>> config = CorrelationConfig(lookback_days=60, alert_threshold=0.75)
        >>> calculator = CorrelationCalculator(config)
        >>> returns = calculator.calculate_returns(prices_df)
        >>> matrix = calculator.calculate_correlation_matrix(returns)
        >>> high_corr = calculator.find_high_correlations(matrix, threshold=0.75)
    """

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        """Initialize the correlation calculator.

        Args:
            config: Configuration for correlation calculation. If None, uses default.
        """
        self.config = config if config is not None else CorrelationConfig()

        logger.debug(
            "correlation_calculator_initialized",
            lookback_days=self.config.lookback_days,
            min_observations=self.config.min_observations,
            alert_threshold=self.config.alert_threshold,
        )

    def calculate_returns(self, prices_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate returns from price data.

        This method computes log returns for each column (symbol) in the prices DataFrame.
        The first column is assumed to be a timestamp and is preserved. All other columns
        are treated as price series.

        Args:
            prices_df: DataFrame with timestamp column and price columns for each symbol.

        Returns:
            DataFrame with timestamp and return columns for each symbol.

        Raises:
            ValueError: If DataFrame is empty or has fewer than 2 columns.
        """
        if prices_df.height == 0:
            raise ValueError("Cannot calculate returns on empty DataFrame")

        if prices_df.width < 2:
            raise ValueError(
                f"DataFrame must have at least 2 columns (timestamp + 1 price), got {prices_df.width}"
            )

        logger.debug("calculating_returns", n_symbols=prices_df.width - 1, n_rows=prices_df.height)

        # Sort by first column (assumed to be timestamp)
        first_col = prices_df.columns[0]
        df_sorted = prices_df.sort(first_col)

        # Calculate log returns for all price columns
        price_cols = df_sorted.columns[1:]
        return_exprs = []

        for col in price_cols:
            return_exprs.append(
                (pl.col(col).log() - pl.col(col).log().shift(1)).alias(f"{col}_return")
            )

        result = df_sorted.select([pl.col(first_col)] + return_exprs)

        logger.debug("returns_calculated", n_symbols=len(price_cols))

        return result

    def calculate_correlation_matrix(
        self, returns_df: pl.DataFrame, method: str = "pearson"
    ) -> pl.DataFrame:
        """Calculate correlation matrix from returns data.

        Args:
            returns_df: DataFrame with return columns for each symbol.
                First column is assumed to be timestamp and is ignored.
            method: Correlation method to use. Either "pearson" or "spearman".
                Default is "pearson".

        Returns:
            DataFrame with correlation matrix. Rows and columns are labeled with
            symbol names (without "_return" suffix).

        Raises:
            ValueError: If DataFrame is invalid or has insufficient observations.
        """
        if returns_df.height == 0:
            raise ValueError("Cannot calculate correlation on empty DataFrame")

        if returns_df.width < 2:
            raise ValueError(
                f"DataFrame must have at least 1 return column, got {returns_df.width - 1}"
            )

        if method not in ["pearson", "spearman"]:
            raise ValueError(f"method must be 'pearson' or 'spearman', got '{method}'")

        logger.debug(
            "calculating_correlation_matrix",
            n_symbols=returns_df.width - 1,
            n_observations=returns_df.height,
            method=method,
        )

        # Get return columns (all except first which is timestamp)
        return_cols = returns_df.columns[1:]

        # Extract symbol names (remove "_return" suffix)
        symbols = [col.replace("_return", "") for col in return_cols]

        # Drop rows with any null values
        returns_clean = returns_df.select(return_cols).drop_nulls()

        if returns_clean.height < self.config.min_observations:
            raise ValueError(
                f"Insufficient observations: need at least {self.config.min_observations}, "
                f"got {returns_clean.height} after removing nulls"
            )

        # Calculate correlation matrix
        n_symbols = len(return_cols)
        corr_matrix = np.zeros((n_symbols, n_symbols))

        for i, col_i in enumerate(return_cols):
            for j, col_j in enumerate(return_cols):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    if method == "pearson":
                        corr = pearson_correlation(returns_clean[col_i], returns_clean[col_j])
                    else:
                        corr = spearman_correlation(returns_clean[col_i], returns_clean[col_j])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        # Create DataFrame with symbol names
        result = pl.DataFrame(corr_matrix, schema=dict.fromkeys(symbols, pl.Float64))

        logger.debug("correlation_matrix_calculated", n_symbols=n_symbols, method=method)

        return result

    def calculate_rolling_correlation(
        self, returns_df: pl.DataFrame, window: int
    ) -> dict[str, pl.Series]:
        """Calculate rolling correlation for each symbol pair.

        Args:
            returns_df: DataFrame with return columns for each symbol.
            window: Rolling window size in number of observations.

        Returns:
            Dictionary mapping symbol pair names (e.g., "AAPL_MSFT") to rolling
            correlation Series.

        Raises:
            ValueError: If window size is invalid or insufficient data.
        """
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")

        if returns_df.height < window:
            raise ValueError(
                f"Insufficient data: need at least {window} rows, got {returns_df.height}"
            )

        logger.debug(
            "calculating_rolling_correlation",
            n_symbols=returns_df.width - 1,
            window=window,
            n_observations=returns_df.height,
        )

        return_cols = returns_df.columns[1:]
        symbols = [col.replace("_return", "") for col in return_cols]

        rolling_corrs: dict[str, pl.Series] = {}

        # Calculate rolling correlation for each pair
        for i in range(len(return_cols)):
            for j in range(i + 1, len(return_cols)):
                col_i = return_cols[i]
                col_j = return_cols[j]
                pair_name = f"{symbols[i]}_{symbols[j]}"

                # Calculate rolling correlation using Polars
                rolling_values: list[float | None] = []

                for start_idx in range(returns_df.height - window + 1):
                    end_idx = start_idx + window
                    window_df = returns_df[start_idx:end_idx]

                    # Drop nulls for this window
                    clean_df = window_df.select([col_i, col_j]).drop_nulls()

                    if clean_df.height >= self.config.min_observations:
                        try:
                            corr = pearson_correlation(clean_df[col_i], clean_df[col_j])
                            rolling_values.append(corr)
                        except ValueError:
                            rolling_values.append(None)
                    else:
                        rolling_values.append(None)

                # Pad with None for the first (window - 1) values
                padding: list[float | None] = [None] * (window - 1)
                padded_values: list[float | None] = padding + rolling_values
                rolling_corrs[pair_name] = pl.Series(pair_name, padded_values)

        logger.debug("rolling_correlation_calculated", n_pairs=len(rolling_corrs))

        return rolling_corrs

    def find_high_correlations(
        self, matrix: pl.DataFrame, threshold: float
    ) -> list[tuple[str, str, float]]:
        """Find symbol pairs with correlation above threshold.

        Args:
            matrix: Correlation matrix DataFrame.
            threshold: Absolute correlation threshold for alerts.

        Returns:
            List of tuples (symbol1, symbol2, correlation) where absolute correlation
            exceeds threshold. Each pair appears only once.

        Raises:
            ValueError: If threshold is invalid.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

        logger.debug("finding_high_correlations", threshold=threshold, n_symbols=matrix.width)

        high_corrs: list[tuple[str, str, float]] = []
        symbols = matrix.columns

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:  # Only check upper triangle to avoid duplicates
                    corr = float(matrix[symbol2][i])  # Access [column][row]
                    if abs(corr) > threshold:
                        high_corrs.append((symbol1, symbol2, corr))

        # Sort by absolute correlation descending
        high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

        logger.info(
            "high_correlations_found",
            n_high_correlations=len(high_corrs),
            threshold=threshold,
        )

        return high_corrs

    def detect_correlation_breakdown(
        self,
        current_matrix: pl.DataFrame,
        previous_matrix: pl.DataFrame,
        threshold: float,
        regime: str | None = None,
    ) -> list[CorrelationBreakdown]:
        """Detect significant changes in correlation between two periods.

        Args:
            current_matrix: Correlation matrix for current period.
            previous_matrix: Correlation matrix for previous period.
            threshold: Minimum absolute change in correlation to trigger breakdown alert.
            regime: Optional market regime label for the current period.

        Returns:
            List of CorrelationBreakdown objects describing significant changes.

        Raises:
            ValueError: If matrices have different dimensions or symbols.
        """
        if current_matrix.width != previous_matrix.width:
            raise ValueError(
                f"Matrices must have same dimensions: {current_matrix.width} != {previous_matrix.width}"
            )

        if current_matrix.columns != previous_matrix.columns:
            raise ValueError("Matrices must have same symbols in same order")

        logger.debug(
            "detecting_correlation_breakdown",
            threshold=threshold,
            n_symbols=current_matrix.width,
            regime=regime,
        )

        breakdowns: list[CorrelationBreakdown] = []
        symbols = current_matrix.columns

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:  # Only check upper triangle
                    current_corr = float(current_matrix[symbol2][i])
                    previous_corr = float(previous_matrix[symbol2][i])
                    change = current_corr - previous_corr

                    if abs(change) >= threshold:
                        breakdown = CorrelationBreakdown(
                            symbol_pair=(symbol1, symbol2),
                            previous_corr=previous_corr,
                            current_corr=current_corr,
                            change=change,
                            regime=regime,
                        )
                        breakdowns.append(breakdown)

        # Sort by absolute change descending
        breakdowns.sort(key=lambda x: abs(x.change), reverse=True)

        logger.info(
            "correlation_breakdowns_detected",
            n_breakdowns=len(breakdowns),
            threshold=threshold,
            regime=regime,
        )

        return breakdowns

    def calculate_diversification_ratio(
        self, matrix: pl.DataFrame, weights: list[float]
    ) -> float:
        """Calculate portfolio diversification ratio.

        The diversification ratio is the ratio of the weighted average volatility
        to the portfolio volatility. A higher ratio indicates better diversification.
        This implementation uses correlation structure to estimate the ratio.

        Args:
            matrix: Correlation matrix DataFrame.
            weights: Portfolio weights for each symbol (must sum to 1.0).

        Returns:
            Diversification ratio.

        Raises:
            ValueError: If weights are invalid or don't match matrix dimensions.
        """
        if len(weights) != matrix.width:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match matrix dimension ({matrix.width})"
            )

        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative")

        logger.debug("calculating_diversification_ratio", n_symbols=len(weights))

        # Convert to numpy for easier matrix operations
        corr_np = matrix.to_numpy()
        weights_np = np.array(weights)

        # Assume equal volatility for all assets (since we only have correlation matrix)
        # In practice, you would pass actual volatilities
        volatilities = np.ones(len(weights))

        # Portfolio variance = w^T * Corr * w (assuming unit volatilities)
        portfolio_variance = weights_np @ corr_np @ weights_np
        portfolio_vol = np.sqrt(portfolio_variance)

        # Weighted average volatility
        weighted_avg_vol = np.sum(weights_np * volatilities)

        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        logger.debug("diversification_ratio_calculated", ratio=div_ratio)

        return float(div_ratio)

    def get_correlation_clusters(
        self, matrix: pl.DataFrame, n_clusters: int, method: str = "average"
    ) -> dict[int, list[str]]:
        """Cluster symbols based on correlation structure using hierarchical clustering.

        Args:
            matrix: Correlation matrix DataFrame.
            n_clusters: Number of clusters to create.
            method: Linkage method for hierarchical clustering. Options: 'single',
                'complete', 'average', 'weighted', 'centroid', 'median', 'ward'.
                Default is 'average'.

        Returns:
            Dictionary mapping cluster ID to list of symbol names.

        Raises:
            ValueError: If n_clusters is invalid or method is unsupported.
        """
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")

        if n_clusters > matrix.width:
            raise ValueError(
                f"n_clusters ({n_clusters}) cannot exceed number of symbols ({matrix.width})"
            )

        valid_methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        logger.debug(
            "clustering_correlations",
            n_symbols=matrix.width,
            n_clusters=n_clusters,
            method=method,
        )

        # Convert correlation to distance (1 - correlation)
        corr_np = matrix.to_numpy()
        distance_matrix = 1 - corr_np

        # Ensure the matrix is symmetric and non-negative
        distance_matrix = np.clip(distance_matrix, 0, 2)

        # Convert to condensed distance matrix (upper triangle)
        condensed_dist = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method=method)

        # Cut dendrogram to get clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

        # Group symbols by cluster
        symbols = matrix.columns
        clusters: dict[int, list[str]] = {}

        for symbol, cluster_id in zip(symbols, cluster_labels, strict=True):
            cluster_id = int(cluster_id)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(symbol)

        logger.info(
            "correlation_clusters_created",
            n_clusters=len(clusters),
            cluster_sizes={k: len(v) for k, v in clusters.items()},
        )

        return clusters
