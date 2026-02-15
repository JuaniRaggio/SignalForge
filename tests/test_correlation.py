"""Tests for correlation matrix calculation and analysis.

This module tests the CorrelationCalculator class including initialization,
configuration validation, correlation calculation, breakdown detection, and
diversification metrics.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalforge.risk.correlation import (
    CorrelationBreakdown,
    CorrelationCalculator,
    CorrelationConfig,
    CorrelationResult,
    pearson_correlation,
    spearman_correlation,
)


@pytest.fixture
def sample_prices() -> pl.DataFrame:
    """Create sample price data for testing.

    Returns a DataFrame with 100 rows of synthetic price data
    for 4 different symbols with varying correlation patterns.
    """
    n_rows = 100
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    np.random.seed(42)

    # Create correlated price series
    base = np.cumsum(np.random.randn(n_rows) * 0.02)
    aapl = 100.0 + base + np.random.randn(n_rows) * 0.01
    msft = 200.0 + base * 1.2 + np.random.randn(n_rows) * 0.015  # Highly correlated with AAPL
    googl = 150.0 + base * 0.5 + np.random.randn(n_rows) * 0.02  # Moderately correlated
    tsla = 300.0 + np.cumsum(np.random.randn(n_rows) * 0.03)  # Low correlation

    return pl.DataFrame(
        {
            "timestamp": dates,
            "AAPL": aapl,
            "MSFT": msft,
            "GOOGL": googl,
            "TSLA": tsla,
        }
    )


@pytest.fixture
def sample_returns() -> pl.DataFrame:
    """Create sample returns data for testing."""
    n_rows = 100
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    np.random.seed(42)

    # Create return series with known correlations
    common_factor = np.random.randn(n_rows) * 0.02
    aapl_ret = common_factor + np.random.randn(n_rows) * 0.01
    msft_ret = common_factor * 0.9 + np.random.randn(n_rows) * 0.01
    googl_ret = common_factor * 0.5 + np.random.randn(n_rows) * 0.015
    tsla_ret = np.random.randn(n_rows) * 0.03

    return pl.DataFrame(
        {
            "timestamp": dates,
            "AAPL_return": aapl_ret,
            "MSFT_return": msft_ret,
            "GOOGL_return": googl_ret,
            "TSLA_return": tsla_ret,
        }
    )


class TestCorrelationConfig:
    """Tests for CorrelationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration initialization."""
        config = CorrelationConfig()

        assert config.lookback_days == 60
        assert config.min_observations == 20
        assert config.alert_threshold == 0.7

    def test_custom_config(self) -> None:
        """Test custom configuration initialization."""
        config = CorrelationConfig(
            lookback_days=90, min_observations=30, alert_threshold=0.8
        )

        assert config.lookback_days == 90
        assert config.min_observations == 30
        assert config.alert_threshold == 0.8

    def test_config_validation_lookback_days(self) -> None:
        """Test validation for lookback_days."""
        with pytest.raises(ValueError, match="lookback_days must be at least 2"):
            CorrelationConfig(lookback_days=1)

    def test_config_validation_min_observations(self) -> None:
        """Test validation for min_observations."""
        with pytest.raises(ValueError, match="min_observations must be at least 2"):
            CorrelationConfig(min_observations=1)

    def test_config_validation_min_obs_exceeds_lookback(self) -> None:
        """Test validation for min_observations > lookback_days."""
        with pytest.raises(ValueError, match="min_observations .* cannot exceed lookback_days"):
            CorrelationConfig(lookback_days=30, min_observations=50)

    def test_config_validation_alert_threshold_negative(self) -> None:
        """Test validation for negative alert_threshold."""
        with pytest.raises(ValueError, match="alert_threshold must be between 0.0 and 1.0"):
            CorrelationConfig(alert_threshold=-0.1)

    def test_config_validation_alert_threshold_too_high(self) -> None:
        """Test validation for alert_threshold > 1.0."""
        with pytest.raises(ValueError, match="alert_threshold must be between 0.0 and 1.0"):
            CorrelationConfig(alert_threshold=1.5)


class TestPearsonCorrelation:
    """Tests for pearson_correlation function."""

    def test_perfect_positive_correlation(self) -> None:
        """Test perfect positive correlation (r = 1.0)."""
        x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series("y", [2.0, 4.0, 6.0, 8.0, 10.0])

        corr = pearson_correlation(x, y)
        assert abs(corr - 1.0) < 1e-10

    def test_perfect_negative_correlation(self) -> None:
        """Test perfect negative correlation (r = -1.0)."""
        x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series("y", [10.0, 8.0, 6.0, 4.0, 2.0])

        corr = pearson_correlation(x, y)
        assert abs(corr - (-1.0)) < 1e-10

    def test_no_correlation(self) -> None:
        """Test no correlation (r close to 0)."""
        np.random.seed(42)
        x = pl.Series("x", np.random.randn(1000))
        y = pl.Series("y", np.random.randn(1000))

        corr = pearson_correlation(x, y)
        assert abs(corr) < 0.1  # Should be close to 0

    def test_constant_series(self) -> None:
        """Test with constant series (undefined correlation)."""
        x = pl.Series("x", [1.0, 1.0, 1.0, 1.0, 1.0])
        y = pl.Series("y", [2.0, 4.0, 6.0, 8.0, 10.0])

        corr = pearson_correlation(x, y)
        assert corr == 0.0

    def test_with_nulls(self) -> None:
        """Test correlation with null values."""
        x = pl.Series("x", [1.0, 2.0, None, 4.0, 5.0])
        y = pl.Series("y", [2.0, 4.0, 6.0, 8.0, 10.0])

        corr = pearson_correlation(x, y)
        # Should calculate on valid pairs only
        assert abs(corr - 1.0) < 1e-10

    def test_different_lengths_raises(self) -> None:
        """Test that different length series raise ValueError."""
        x = pl.Series("x", [1.0, 2.0, 3.0])
        y = pl.Series("y", [1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="Series must have same length"):
            pearson_correlation(x, y)

    def test_all_nulls_raises(self) -> None:
        """Test that all null values raise ValueError."""
        x = pl.Series("x", [None, None, None])
        y = pl.Series("y", [None, None, None])

        with pytest.raises(ValueError, match="No valid observations"):
            pearson_correlation(x, y)


class TestSpearmanCorrelation:
    """Tests for spearman_correlation function."""

    def test_perfect_monotonic_positive(self) -> None:
        """Test perfect monotonic positive relationship."""
        x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series("y", [1.0, 4.0, 9.0, 16.0, 25.0])  # y = x^2

        corr = spearman_correlation(x, y)
        assert abs(corr - 1.0) < 1e-10

    def test_perfect_monotonic_negative(self) -> None:
        """Test perfect monotonic negative relationship."""
        x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series("y", [25.0, 16.0, 9.0, 4.0, 1.0])

        corr = spearman_correlation(x, y)
        assert abs(corr - (-1.0)) < 1e-10

    def test_robust_to_outliers(self) -> None:
        """Test that Spearman is more robust to outliers than Pearson."""
        x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series("y", [2.0, 4.0, 6.0, 8.0, 100.0])  # Outlier at end

        pearson = pearson_correlation(x, y)
        spearman = spearman_correlation(x, y)

        # Spearman should be closer to 1 than Pearson due to outlier
        assert abs(spearman - 1.0) < abs(pearson - 1.0)

    def test_constant_series(self) -> None:
        """Test with constant series."""
        x = pl.Series("x", [1.0, 1.0, 1.0, 1.0, 1.0])
        y = pl.Series("y", [2.0, 4.0, 6.0, 8.0, 10.0])

        corr = spearman_correlation(x, y)
        assert corr == 0.0

    def test_with_nulls(self) -> None:
        """Test correlation with null values."""
        x = pl.Series("x", [1.0, 2.0, None, 4.0, 5.0])
        y = pl.Series("y", [2.0, 4.0, 6.0, 8.0, 10.0])

        corr = spearman_correlation(x, y)
        # Should calculate on valid pairs only
        assert abs(corr - 1.0) < 1e-10

    def test_different_lengths_raises(self) -> None:
        """Test that different length series raise ValueError."""
        x = pl.Series("x", [1.0, 2.0, 3.0])
        y = pl.Series("y", [1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="Series must have same length"):
            spearman_correlation(x, y)

    def test_all_nulls_raises(self) -> None:
        """Test that all null values raise ValueError."""
        x = pl.Series("x", [None, None, None])
        y = pl.Series("y", [None, None, None])

        with pytest.raises(ValueError, match="No valid observations"):
            spearman_correlation(x, y)


class TestCalculatorInitialization:
    """Tests for CorrelationCalculator initialization."""

    def test_initialization_default_config(self) -> None:
        """Test initialization with default configuration."""
        calculator = CorrelationCalculator()

        assert calculator.config.lookback_days == 60
        assert calculator.config.min_observations == 20
        assert calculator.config.alert_threshold == 0.7

    def test_initialization_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = CorrelationConfig(lookback_days=90, alert_threshold=0.8)
        calculator = CorrelationCalculator(config=config)

        assert calculator.config.lookback_days == 90
        assert calculator.config.alert_threshold == 0.8


class TestCalculateReturns:
    """Tests for calculate_returns method."""

    def test_calculate_returns_basic(self, sample_prices: pl.DataFrame) -> None:
        """Test basic returns calculation."""
        calculator = CorrelationCalculator()
        returns = calculator.calculate_returns(sample_prices)

        # Check structure
        assert "timestamp" in returns.columns
        assert "AAPL_return" in returns.columns
        assert "MSFT_return" in returns.columns
        assert "GOOGL_return" in returns.columns
        assert "TSLA_return" in returns.columns
        assert returns.height == sample_prices.height

    def test_calculate_returns_values(self) -> None:
        """Test that returns are calculated correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
                "AAPL": [100.0, 105.0, 110.0],
            }
        )

        calculator = CorrelationCalculator()
        returns = calculator.calculate_returns(df)

        # First return should be null
        assert returns["AAPL_return"][0] is None

        # Calculate expected log returns
        expected_ret_1 = np.log(105.0 / 100.0)
        expected_ret_2 = np.log(110.0 / 105.0)

        assert abs(returns["AAPL_return"][1] - expected_ret_1) < 1e-10
        assert abs(returns["AAPL_return"][2] - expected_ret_2) < 1e-10

    def test_calculate_returns_empty_df(self) -> None:
        """Test with empty DataFrame."""
        df = pl.DataFrame({"timestamp": [], "AAPL": []})
        calculator = CorrelationCalculator()

        with pytest.raises(ValueError, match="Cannot calculate returns on empty DataFrame"):
            calculator.calculate_returns(df)

    def test_calculate_returns_single_column(self) -> None:
        """Test with only timestamp column."""
        df = pl.DataFrame({"timestamp": [datetime(2023, 1, 1)]})
        calculator = CorrelationCalculator()

        with pytest.raises(ValueError, match="DataFrame must have at least 2 columns"):
            calculator.calculate_returns(df)


class TestCalculateCorrelationMatrix:
    """Tests for calculate_correlation_matrix method."""

    def test_correlation_matrix_basic(self, sample_returns: pl.DataFrame) -> None:
        """Test basic correlation matrix calculation."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        # Check structure
        assert matrix.width == 4  # 4 symbols
        assert matrix.height == 4
        assert "AAPL" in matrix.columns
        assert "MSFT" in matrix.columns

    def test_correlation_matrix_diagonal_ones(self, sample_returns: pl.DataFrame) -> None:
        """Test that diagonal elements are 1.0."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        symbols = matrix.columns
        for i, symbol in enumerate(symbols):
            assert abs(matrix[symbol][i] - 1.0) < 1e-10

    def test_correlation_matrix_symmetric(self, sample_returns: pl.DataFrame) -> None:
        """Test that correlation matrix is symmetric."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        corr_np = matrix.to_numpy()
        assert np.allclose(corr_np, corr_np.T)

    def test_correlation_matrix_pearson(self, sample_returns: pl.DataFrame) -> None:
        """Test Pearson correlation method."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns, method="pearson")

        # All values should be in [-1, 1]
        corr_np = matrix.to_numpy()
        assert np.all(corr_np >= -1.0)
        assert np.all(corr_np <= 1.0)

    def test_correlation_matrix_spearman(self, sample_returns: pl.DataFrame) -> None:
        """Test Spearman correlation method."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns, method="spearman")

        # All values should be in [-1, 1]
        corr_np = matrix.to_numpy()
        assert np.all(corr_np >= -1.0)
        assert np.all(corr_np <= 1.0)

    def test_correlation_matrix_invalid_method(self, sample_returns: pl.DataFrame) -> None:
        """Test with invalid correlation method."""
        calculator = CorrelationCalculator()

        with pytest.raises(ValueError, match="method must be 'pearson' or 'spearman'"):
            calculator.calculate_correlation_matrix(sample_returns, method="kendall")

    def test_correlation_matrix_insufficient_observations(self) -> None:
        """Test with insufficient observations."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "AAPL_return": [0.01, 0.02],
                "MSFT_return": [0.015, 0.025],
            }
        )

        config = CorrelationConfig(min_observations=20)
        calculator = CorrelationCalculator(config=config)

        with pytest.raises(ValueError, match="Insufficient observations"):
            calculator.calculate_correlation_matrix(df)

    def test_correlation_matrix_empty_df(self) -> None:
        """Test with empty DataFrame."""
        df = pl.DataFrame({"timestamp": [], "AAPL_return": []})
        calculator = CorrelationCalculator()

        with pytest.raises(ValueError, match="Cannot calculate correlation on empty DataFrame"):
            calculator.calculate_correlation_matrix(df)


class TestCalculateRollingCorrelation:
    """Tests for calculate_rolling_correlation method."""

    def test_rolling_correlation_basic(self, sample_returns: pl.DataFrame) -> None:
        """Test basic rolling correlation calculation."""
        calculator = CorrelationCalculator()
        rolling = calculator.calculate_rolling_correlation(sample_returns, window=20)

        # Check that we have correlations for each pair
        assert "AAPL_MSFT" in rolling
        assert "AAPL_GOOGL" in rolling
        assert "MSFT_GOOGL" in rolling
        assert len(rolling) == 6  # C(4, 2) = 6 pairs

    def test_rolling_correlation_length(self, sample_returns: pl.DataFrame) -> None:
        """Test that rolling correlation series have correct length."""
        calculator = CorrelationCalculator()
        window = 20
        rolling = calculator.calculate_rolling_correlation(sample_returns, window=window)

        for series in rolling.values():
            assert series.len() == sample_returns.height

    def test_rolling_correlation_padding(self, sample_returns: pl.DataFrame) -> None:
        """Test that first (window - 1) values are None."""
        calculator = CorrelationCalculator()
        window = 20
        rolling = calculator.calculate_rolling_correlation(sample_returns, window=window)

        for series in rolling.values():
            # First (window - 1) should be None
            for i in range(window - 1):
                assert series[i] is None

    def test_rolling_correlation_invalid_window(self, sample_returns: pl.DataFrame) -> None:
        """Test with invalid window size."""
        calculator = CorrelationCalculator()

        with pytest.raises(ValueError, match="window must be at least 2"):
            calculator.calculate_rolling_correlation(sample_returns, window=1)

    def test_rolling_correlation_insufficient_data(self) -> None:
        """Test with insufficient data for window."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "AAPL_return": [0.01],
                "MSFT_return": [0.015],
            }
        )

        calculator = CorrelationCalculator()

        with pytest.raises(ValueError, match="Insufficient data"):
            calculator.calculate_rolling_correlation(df, window=20)


class TestFindHighCorrelations:
    """Tests for find_high_correlations method."""

    def test_find_high_correlations_basic(self, sample_returns: pl.DataFrame) -> None:
        """Test finding high correlations."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)
        high_corrs = calculator.find_high_correlations(matrix, threshold=0.5)

        # Should return list of tuples
        assert isinstance(high_corrs, list)
        for item in high_corrs:
            assert len(item) == 3
            symbol1, symbol2, corr = item
            assert isinstance(symbol1, str)
            assert isinstance(symbol2, str)
            assert abs(corr) > 0.5

    def test_find_high_correlations_sorted(self, sample_returns: pl.DataFrame) -> None:
        """Test that results are sorted by absolute correlation descending."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)
        high_corrs = calculator.find_high_correlations(matrix, threshold=0.3)

        if len(high_corrs) > 1:
            for i in range(len(high_corrs) - 1):
                assert abs(high_corrs[i][2]) >= abs(high_corrs[i + 1][2])

    def test_find_high_correlations_no_duplicates(self, sample_returns: pl.DataFrame) -> None:
        """Test that each pair appears only once."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)
        high_corrs = calculator.find_high_correlations(matrix, threshold=0.0)

        pairs = {tuple(sorted([s1, s2])) for s1, s2, _ in high_corrs}
        assert len(pairs) == len(high_corrs)

    def test_find_high_correlations_threshold_validation(
        self, sample_returns: pl.DataFrame
    ) -> None:
        """Test threshold validation."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            calculator.find_high_correlations(matrix, threshold=1.5)

    def test_find_high_correlations_high_threshold(self, sample_returns: pl.DataFrame) -> None:
        """Test with very high threshold."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)
        high_corrs = calculator.find_high_correlations(matrix, threshold=0.99)

        # Should find very few or no correlations
        assert len(high_corrs) <= 2


class TestDetectCorrelationBreakdown:
    """Tests for detect_correlation_breakdown method."""

    def test_breakdown_detection_basic(self, sample_returns: pl.DataFrame) -> None:
        """Test basic breakdown detection."""
        calculator = CorrelationCalculator()

        # Split data into two periods
        mid_point = sample_returns.height // 2
        previous = sample_returns.head(mid_point)
        current = sample_returns.tail(sample_returns.height - mid_point)

        prev_matrix = calculator.calculate_correlation_matrix(previous)
        curr_matrix = calculator.calculate_correlation_matrix(current)

        breakdowns = calculator.detect_correlation_breakdown(
            curr_matrix, prev_matrix, threshold=0.1
        )

        # Should return list of CorrelationBreakdown objects
        assert isinstance(breakdowns, list)
        for breakdown in breakdowns:
            assert isinstance(breakdown, CorrelationBreakdown)
            assert len(breakdown.symbol_pair) == 2
            assert abs(breakdown.change) >= 0.1

    def test_breakdown_detection_with_regime(self, sample_returns: pl.DataFrame) -> None:
        """Test breakdown detection with regime label."""
        calculator = CorrelationCalculator()

        mid_point = sample_returns.height // 2
        previous = sample_returns.head(mid_point)
        current = sample_returns.tail(sample_returns.height - mid_point)

        prev_matrix = calculator.calculate_correlation_matrix(previous)
        curr_matrix = calculator.calculate_correlation_matrix(current)

        breakdowns = calculator.detect_correlation_breakdown(
            curr_matrix, prev_matrix, threshold=0.1, regime="bull"
        )

        for breakdown in breakdowns:
            assert breakdown.regime == "bull"

    def test_breakdown_detection_sorted(self, sample_returns: pl.DataFrame) -> None:
        """Test that breakdowns are sorted by absolute change descending."""
        calculator = CorrelationCalculator()

        mid_point = sample_returns.height // 2
        previous = sample_returns.head(mid_point)
        current = sample_returns.tail(sample_returns.height - mid_point)

        prev_matrix = calculator.calculate_correlation_matrix(previous)
        curr_matrix = calculator.calculate_correlation_matrix(current)

        breakdowns = calculator.detect_correlation_breakdown(
            curr_matrix, prev_matrix, threshold=0.05
        )

        if len(breakdowns) > 1:
            for i in range(len(breakdowns) - 1):
                assert abs(breakdowns[i].change) >= abs(breakdowns[i + 1].change)

    def test_breakdown_detection_different_dimensions(self) -> None:
        """Test with matrices of different dimensions."""
        calculator = CorrelationCalculator()

        matrix1 = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
        matrix2 = pl.DataFrame({"A": [1.0, 0.5, 0.3], "B": [0.5, 1.0, 0.4], "C": [0.3, 0.4, 1.0]})

        with pytest.raises(ValueError, match="Matrices must have same dimensions"):
            calculator.detect_correlation_breakdown(matrix2, matrix1, threshold=0.1)

    def test_breakdown_detection_different_symbols(self) -> None:
        """Test with matrices having different symbols."""
        calculator = CorrelationCalculator()

        matrix1 = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
        matrix2 = pl.DataFrame({"A": [1.0, 0.5], "C": [0.5, 1.0]})

        with pytest.raises(ValueError, match="Matrices must have same symbols"):
            calculator.detect_correlation_breakdown(matrix2, matrix1, threshold=0.1)


class TestCalculateDiversificationRatio:
    """Tests for calculate_diversification_ratio method."""

    def test_diversification_ratio_equal_weights(self, sample_returns: pl.DataFrame) -> None:
        """Test diversification ratio with equal weights."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        weights = [0.25, 0.25, 0.25, 0.25]
        ratio = calculator.calculate_diversification_ratio(matrix, weights)

        # Ratio should be >= 1.0 for diversified portfolio
        assert ratio >= 1.0

    def test_diversification_ratio_concentrated(self, sample_returns: pl.DataFrame) -> None:
        """Test diversification ratio with concentrated weights."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        weights = [1.0, 0.0, 0.0, 0.0]
        ratio = calculator.calculate_diversification_ratio(matrix, weights)

        # Ratio should be close to 1.0 for concentrated portfolio
        assert abs(ratio - 1.0) < 0.01

    def test_diversification_ratio_invalid_weights_sum(
        self, sample_returns: pl.DataFrame
    ) -> None:
        """Test with weights not summing to 1.0."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        weights = [0.3, 0.3, 0.3, 0.3]

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            calculator.calculate_diversification_ratio(matrix, weights)

    def test_diversification_ratio_negative_weights(self, sample_returns: pl.DataFrame) -> None:
        """Test with negative weights."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        weights = [0.5, 0.5, 0.5, -0.5]

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            calculator.calculate_diversification_ratio(matrix, weights)

    def test_diversification_ratio_wrong_length(self, sample_returns: pl.DataFrame) -> None:
        """Test with wrong number of weights."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        weights = [0.5, 0.5]

        with pytest.raises(ValueError, match="Number of weights .* must match matrix dimension"):
            calculator.calculate_diversification_ratio(matrix, weights)


class TestGetCorrelationClusters:
    """Tests for get_correlation_clusters method."""

    def test_clustering_basic(self, sample_returns: pl.DataFrame) -> None:
        """Test basic clustering."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        clusters = calculator.get_correlation_clusters(matrix, n_clusters=2)

        # Should return dict with cluster IDs
        assert isinstance(clusters, dict)
        assert len(clusters) == 2

        # All symbols should be assigned
        all_symbols = []
        for cluster_symbols in clusters.values():
            all_symbols.extend(cluster_symbols)
        assert len(all_symbols) == 4

    def test_clustering_all_in_one(self, sample_returns: pl.DataFrame) -> None:
        """Test clustering with n_clusters=1."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        clusters = calculator.get_correlation_clusters(matrix, n_clusters=1)

        assert len(clusters) == 1
        # All symbols should be in one cluster
        all_symbols = list(clusters.values())[0]
        assert len(all_symbols) == 4

    def test_clustering_each_separate(self, sample_returns: pl.DataFrame) -> None:
        """Test clustering with n_clusters equal to number of symbols."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        clusters = calculator.get_correlation_clusters(matrix, n_clusters=4)

        assert len(clusters) <= 4
        # Each cluster should have at least one symbol
        for cluster_symbols in clusters.values():
            assert len(cluster_symbols) >= 1

    def test_clustering_invalid_n_clusters(self, sample_returns: pl.DataFrame) -> None:
        """Test with invalid n_clusters."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        with pytest.raises(ValueError, match="n_clusters must be at least 1"):
            calculator.get_correlation_clusters(matrix, n_clusters=0)

        with pytest.raises(ValueError, match="n_clusters .* cannot exceed number of symbols"):
            calculator.get_correlation_clusters(matrix, n_clusters=10)

    def test_clustering_invalid_method(self, sample_returns: pl.DataFrame) -> None:
        """Test with invalid clustering method."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        with pytest.raises(ValueError, match="method must be one of"):
            calculator.get_correlation_clusters(matrix, n_clusters=2, method="invalid")

    def test_clustering_different_methods(self, sample_returns: pl.DataFrame) -> None:
        """Test different clustering methods."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        methods = ["single", "complete", "average", "weighted"]
        for method in methods:
            clusters = calculator.get_correlation_clusters(matrix, n_clusters=2, method=method)
            assert len(clusters) <= 2


class TestCorrelationBreakdownDataclass:
    """Tests for CorrelationBreakdown dataclass."""

    def test_breakdown_creation(self) -> None:
        """Test creating CorrelationBreakdown object."""
        breakdown = CorrelationBreakdown(
            symbol_pair=("AAPL", "MSFT"),
            previous_corr=0.8,
            current_corr=0.5,
            change=-0.3,
        )

        assert breakdown.symbol_pair == ("AAPL", "MSFT")
        assert breakdown.previous_corr == 0.8
        assert breakdown.current_corr == 0.5
        assert breakdown.change == -0.3
        assert breakdown.regime is None

    def test_breakdown_with_regime(self) -> None:
        """Test creating CorrelationBreakdown with regime."""
        breakdown = CorrelationBreakdown(
            symbol_pair=("AAPL", "MSFT"),
            previous_corr=0.8,
            current_corr=0.5,
            change=-0.3,
            regime="bear",
        )

        assert breakdown.regime == "bear"


class TestCorrelationResultDataclass:
    """Tests for CorrelationResult dataclass."""

    def test_result_creation(self, sample_returns: pl.DataFrame) -> None:
        """Test creating CorrelationResult object."""
        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(sample_returns)

        result = CorrelationResult(
            matrix=matrix,
            high_correlations=[("AAPL", "MSFT", 0.85)],
            avg_correlation=0.45,
            timestamp=datetime.now(),
        )

        assert isinstance(result.matrix, pl.DataFrame)
        assert len(result.high_correlations) == 1
        assert result.avg_correlation == 0.45
        assert isinstance(result.timestamp, datetime)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_two_symbol_correlation(self) -> None:
        """Test correlation with only two symbols."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)],
                "AAPL_return": np.random.randn(50) * 0.02,
                "MSFT_return": np.random.randn(50) * 0.02,
            }
        )

        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(df)

        assert matrix.width == 2
        assert matrix.height == 2

    def test_with_nan_values(self) -> None:
        """Test correlation calculation with NaN values."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)],
                "AAPL_return": [0.01 if i % 5 != 0 else None for i in range(50)],
                "MSFT_return": [0.015 if i % 7 != 0 else None for i in range(50)],
            }
        )

        calculator = CorrelationCalculator(config=CorrelationConfig(min_observations=20))
        matrix = calculator.calculate_correlation_matrix(df)

        # Should handle NaN gracefully
        assert matrix.width == 2
        assert matrix.height == 2

    def test_perfect_correlation_clustering(self) -> None:
        """Test clustering with perfectly correlated symbols."""
        # Create perfectly correlated returns
        returns = np.random.randn(50) * 0.02
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)],
                "A_return": returns,
                "B_return": returns,
                "C_return": returns,
            }
        )

        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(df)

        # All should cluster together
        clusters = calculator.get_correlation_clusters(matrix, n_clusters=1)
        assert len(clusters[1]) == 3

    def test_zero_volatility(self) -> None:
        """Test with zero volatility returns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)],
                "AAPL_return": [0.0] * 50,
                "MSFT_return": np.random.randn(50) * 0.02,
            }
        )

        calculator = CorrelationCalculator()
        matrix = calculator.calculate_correlation_matrix(df)

        # Should handle zero volatility gracefully
        assert matrix.width == 2
        # Correlation with constant series should be 0
        assert matrix["MSFT"][0] == 0.0
