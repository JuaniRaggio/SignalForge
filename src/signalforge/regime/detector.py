"""Hidden Markov Model based market regime detector.

This module implements regime detection using Gaussian Hidden Markov Models
to identify different market states. The detector analyzes price and volume
data to classify market periods into distinct regimes.

The implementation uses multiple features to capture different aspects of
market behavior:
- Log returns: Capture price direction and magnitude
- Volatility: Measure of market uncertainty
- Volume changes: Trading activity patterns
- Trend indicators: Direction and strength of trends

Examples:
    Basic usage with default configuration:

    >>> import polars as pl
    >>> from signalforge.regime.detector import RegimeDetector, RegimeConfig
    >>>
    >>> df = pl.DataFrame({
    ...     "timestamp": pl.date_range(start="2023-01-01", periods=252, interval="1d"),
    ...     "close": [100.0 + i * 0.5 for i in range(252)],
    ...     "volume": [1000000] * 252,
    ... })
    >>> detector = RegimeDetector()
    >>> detector.fit(df)
    >>> predictions = detector.predict(df)
    >>> current_regime = detector.get_current_regime()

    Custom configuration:

    >>> config = RegimeConfig(
    ...     n_regimes=3,
    ...     lookback_window=100,
    ...     min_regime_duration=3,
    ... )
    >>> detector = RegimeDetector(config=config)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


class Regime(str, Enum):
    """Market regime types.

    Attributes:
        BULL: Upward trending market with positive returns and moderate volatility
        BEAR: Downward trending market with negative returns and increasing volatility
        RANGE: Sideways market with low returns and low volatility
        CRISIS: High volatility period with sharp declines and extreme movements
    """

    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"
    CRISIS = "crisis"


@dataclass
class RegimeConfig:
    """Configuration for regime detection.

    Attributes:
        n_regimes: Number of distinct regimes to detect. Default is 4.
        lookback_window: Number of historical periods for training. Default is 252 (1 year).
        min_regime_duration: Minimum number of periods a regime should persist. Default is 5.
        volatility_window: Window size for volatility calculation. Default is 20.
        trend_fast_window: Fast moving average window for trend detection. Default is 10.
        trend_slow_window: Slow moving average window for trend detection. Default is 50.
        random_state: Random seed for reproducibility. Default is 42.
        n_iter: Maximum number of HMM training iterations. Default is 100.
        tol: Convergence tolerance for HMM training. Default is 0.01.
        covariance_type: Type of covariance matrix (spherical, diag, full, tied). Default is diag.
    """

    n_regimes: int = 4
    lookback_window: int = 252
    min_regime_duration: int = 5
    volatility_window: int = 20
    trend_fast_window: int = 10
    trend_slow_window: int = 50
    random_state: int = 42
    n_iter: int = 100
    tol: float = 0.01
    covariance_type: str = "diag"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_regimes < 2:
            raise ValueError(f"n_regimes must be at least 2, got {self.n_regimes}")
        if self.n_regimes > 10:
            raise ValueError(f"n_regimes must be at most 10, got {self.n_regimes}")
        if self.lookback_window < self.volatility_window * 2:
            raise ValueError(
                f"lookback_window ({self.lookback_window}) must be at least "
                f"2x volatility_window ({self.volatility_window})"
            )
        if self.min_regime_duration < 1:
            raise ValueError(
                f"min_regime_duration must be positive, got {self.min_regime_duration}"
            )
        if self.volatility_window < 2:
            raise ValueError(f"volatility_window must be at least 2, got {self.volatility_window}")
        if self.trend_fast_window >= self.trend_slow_window:
            raise ValueError(
                f"trend_fast_window ({self.trend_fast_window}) must be less than "
                f"trend_slow_window ({self.trend_slow_window})"
            )
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be positive, got {self.n_iter}")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if self.covariance_type not in ["spherical", "diag", "full", "tied"]:
            raise ValueError(
                f"covariance_type must be one of spherical, diag, full, tied; "
                f"got {self.covariance_type}"
            )


class RegimeDetector:
    """Hidden Markov Model based market regime detector.

    This class uses a Gaussian HMM to identify distinct market regimes
    from historical price and volume data. It extracts multiple features
    and trains a model to classify periods into different market states.

    The detector maintains a mapping between HMM states and interpretable
    regime labels (BULL, BEAR, RANGE, CRISIS) based on the characteristics
    of each state.

    Attributes:
        config: Configuration parameters for the detector
        _model: Fitted Gaussian HMM model (None until fit() is called)
        _feature_columns: Names of feature columns used for training
        _regime_mapping: Mapping from HMM states to Regime labels
        _training_data: Last training DataFrame for reference
        _fitted: Boolean flag indicating if model is trained
        _scaler_mean: Feature means for standardization
        _scaler_std: Feature standard deviations for standardization

    Examples:
        Fitting and predicting regimes:

        >>> import polars as pl
        >>> from signalforge.regime.detector import RegimeDetector
        >>>
        >>> df = pl.DataFrame({
        ...     "timestamp": pl.date_range(start="2023-01-01", periods=300, interval="1d"),
        ...     "close": [100.0 + i * 0.3 + np.random.randn() * 2 for i in range(300)],
        ...     "volume": [1000000 + i * 100 for i in range(300)],
        ... })
        >>>
        >>> detector = RegimeDetector()
        >>> detector.fit(df)
        >>> predictions = detector.predict(df)
        >>> current_regime = detector.get_current_regime()
        >>> probabilities = detector.get_regime_probabilities()
        >>> transition_matrix = detector.get_transition_matrix()
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        """Initialize the regime detector.

        Args:
            config: Configuration for the detector. If None, uses default configuration.
        """
        self.config = config if config is not None else RegimeConfig()
        self._model: GaussianHMM | None = None
        self._feature_columns: list[str] = []
        self._regime_mapping: dict[int, Regime] = {}
        self._training_data: pl.DataFrame | None = None
        self._fitted: bool = False
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._last_predictions: np.ndarray | None = None

        logger.debug(
            "regime_detector_initialized",
            n_regimes=self.config.n_regimes,
            lookback_window=self.config.lookback_window,
            min_regime_duration=self.config.min_regime_duration,
        )

    def _compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute features for regime detection.

        Args:
            df: Input DataFrame with price data.

        Returns:
            DataFrame with computed features.

        Raises:
            ValueError: If required columns are missing.
        """
        required_cols = ["timestamp", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.debug("computing_regime_features", n_rows=df.height)

        # Sort by timestamp
        df_sorted = df.sort("timestamp")

        # Compute log returns
        df_features = df_sorted.with_columns(
            (pl.col("close").log() - pl.col("close").log().shift(1)).alias("log_return")
        )

        # Compute rolling volatility (std of returns)
        df_features = df_features.with_columns(
            pl.col("log_return")
            .rolling_std(window_size=self.config.volatility_window)
            .alias("volatility")
        )

        # Compute volume changes if volume is available
        if "volume" in df.columns:
            df_features = df_features.with_columns(
                (pl.col("volume").log() - pl.col("volume").log().shift(1)).alias("volume_change")
            )
        else:
            # Use zeros if volume not available
            df_features = df_features.with_columns(pl.lit(0.0).alias("volume_change"))

        # Compute trend indicators (SMA crossover)
        df_features = df_features.with_columns(
            [
                pl.col("close")
                .rolling_mean(window_size=self.config.trend_fast_window)
                .alias("sma_fast"),
                pl.col("close")
                .rolling_mean(window_size=self.config.trend_slow_window)
                .alias("sma_slow"),
            ]
        )

        # Compute trend strength (normalized difference)
        df_features = df_features.with_columns(
            ((pl.col("sma_fast") - pl.col("sma_slow")) / pl.col("close")).alias("trend_strength")
        )

        # Drop intermediate columns
        df_features = df_features.drop(["sma_fast", "sma_slow"])

        # Store feature column names
        self._feature_columns = ["log_return", "volatility", "volume_change", "trend_strength"]

        logger.debug("regime_features_computed", features=self._feature_columns)

        return df_features

    def _prepare_feature_matrix(self, df: pl.DataFrame) -> np.ndarray:
        """Prepare feature matrix for HMM training/prediction.

        Args:
            df: DataFrame with computed features.

        Returns:
            NumPy array of features with shape (n_samples, n_features).

        Raises:
            ValueError: If features contain too many NaN values.
        """
        # Select feature columns
        feature_df = df.select(self._feature_columns)

        # Drop rows with NaN values
        feature_df_clean = feature_df.drop_nulls()

        if feature_df_clean.height == 0:
            raise ValueError("All feature rows contain NaN values after computation")

        # Check if we lost too many rows
        rows_dropped = df.height - feature_df_clean.height
        if rows_dropped > df.height * 0.5:
            logger.warning(
                "many_rows_dropped_due_to_nan",
                rows_dropped=rows_dropped,
                total_rows=df.height,
                percentage=rows_dropped / df.height * 100,
            )

        # Convert to numpy array
        features = feature_df_clean.to_numpy()

        logger.debug("feature_matrix_prepared", shape=features.shape)

        return features

    def _standardize_features(
        self, features: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """Standardize features to zero mean and unit variance.

        Args:
            features: Feature matrix to standardize.
            fit: If True, fit the scaler on the features. If False, use existing scaler.

        Returns:
            Standardized feature matrix.

        Raises:
            RuntimeError: If fit=False but scaler has not been fitted.
        """
        if fit:
            self._scaler_mean = np.mean(features, axis=0)
            self._scaler_std = np.std(features, axis=0)
            # Prevent division by zero
            self._scaler_std = np.where(self._scaler_std == 0, 1.0, self._scaler_std)
            logger.debug("feature_scaler_fitted", mean=self._scaler_mean, std=self._scaler_std)
        else:
            if self._scaler_mean is None or self._scaler_std is None:
                raise RuntimeError("Scaler must be fitted before transform")

        standardized: np.ndarray = (features - self._scaler_mean) / self._scaler_std
        return standardized

    def _map_states_to_regimes(self, features: np.ndarray, states: np.ndarray) -> None:
        """Map HMM states to interpretable regime labels.

        This method analyzes the characteristics of each HMM state and assigns
        a regime label (BULL, BEAR, RANGE, CRISIS) based on average returns,
        volatility, and trend strength.

        Args:
            features: Original (unstandardized) feature matrix.
            states: HMM state sequence.
        """
        logger.debug("mapping_states_to_regimes", n_states=self.config.n_regimes)

        state_characteristics = []

        for state in range(self.config.n_regimes):
            # Get features for this state
            state_mask = states == state
            if not np.any(state_mask):
                # State never appears, assign default values
                state_characteristics.append(
                    {
                        "state": state,
                        "avg_return": 0.0,
                        "avg_volatility": 0.0,
                        "avg_trend": 0.0,
                        "count": 0,
                    }
                )
                continue

            state_features = features[state_mask]

            # Calculate average characteristics
            avg_return = float(np.mean(state_features[:, 0]))  # log_return
            avg_volatility = float(np.mean(state_features[:, 1]))  # volatility
            avg_trend = float(np.mean(state_features[:, 3]))  # trend_strength
            count = int(np.sum(state_mask))

            state_characteristics.append(
                {
                    "state": state,
                    "avg_return": avg_return,
                    "avg_volatility": avg_volatility,
                    "avg_trend": avg_trend,
                    "count": count,
                }
            )

            logger.debug(
                "state_characteristics",
                state=state,
                avg_return=avg_return,
                avg_volatility=avg_volatility,
                avg_trend=avg_trend,
                count=count,
            )

        # Sort states by volatility (highest first) to identify CRISIS
        sorted_by_vol = sorted(state_characteristics, key=lambda x: x["avg_volatility"], reverse=True)

        # Assign regimes based on characteristics
        assigned_regimes: set[Regime] = set()

        for char in sorted_by_vol:
            state_idx = char["state"]
            assert isinstance(state_idx, int)

            # CRISIS: Highest volatility with negative returns
            if (
                Regime.CRISIS not in assigned_regimes
                and char["avg_volatility"] > 0
                and char["avg_return"] < 0
            ):
                self._regime_mapping[state_idx] = Regime.CRISIS
                assigned_regimes.add(Regime.CRISIS)
            # BULL: Positive returns and positive trend
            elif (
                Regime.BULL not in assigned_regimes
                and char["avg_return"] > 0
                and char["avg_trend"] > 0
            ):
                self._regime_mapping[state_idx] = Regime.BULL
                assigned_regimes.add(Regime.BULL)
            # BEAR: Negative returns and negative trend
            elif (
                Regime.BEAR not in assigned_regimes
                and char["avg_return"] < 0
                and char["avg_trend"] < 0
            ):
                self._regime_mapping[state_idx] = Regime.BEAR
                assigned_regimes.add(Regime.BEAR)
            # RANGE: Low volatility with returns close to zero
            elif Regime.RANGE not in assigned_regimes and abs(char["avg_return"]) < 0.001:
                self._regime_mapping[state_idx] = Regime.RANGE
                assigned_regimes.add(Regime.RANGE)
            else:
                # Fallback: assign based on return sign
                if char["avg_return"] > 0 and Regime.BULL not in assigned_regimes:
                    self._regime_mapping[state_idx] = Regime.BULL
                    assigned_regimes.add(Regime.BULL)
                elif char["avg_return"] < 0 and Regime.BEAR not in assigned_regimes:
                    self._regime_mapping[state_idx] = Regime.BEAR
                    assigned_regimes.add(Regime.BEAR)
                else:
                    # Final fallback
                    if Regime.RANGE not in assigned_regimes:
                        self._regime_mapping[state_idx] = Regime.RANGE
                        assigned_regimes.add(Regime.RANGE)
                    elif Regime.BULL not in assigned_regimes:
                        self._regime_mapping[state_idx] = Regime.BULL
                        assigned_regimes.add(Regime.BULL)
                    elif Regime.BEAR not in assigned_regimes:
                        self._regime_mapping[state_idx] = Regime.BEAR
                        assigned_regimes.add(Regime.BEAR)
                    else:
                        self._regime_mapping[state_idx] = Regime.CRISIS
                        assigned_regimes.add(Regime.CRISIS)

        logger.info("regime_mapping_complete", mapping=self._regime_mapping)

    def _apply_minimum_duration_filter(self, regimes: np.ndarray) -> np.ndarray:
        """Apply minimum duration filter to regime predictions.

        This method smooths regime predictions by enforcing that each regime
        must persist for at least min_regime_duration periods. Short regime
        changes are replaced with the surrounding regime.

        Args:
            regimes: Array of regime predictions (as integers).

        Returns:
            Filtered regime array with minimum duration enforced.
        """
        if self.config.min_regime_duration <= 1:
            return regimes

        filtered = regimes.copy()
        n = len(filtered)

        i = 0
        while i < n:
            current_regime = filtered[i]
            duration = 1

            # Count consecutive occurrences
            while i + duration < n and filtered[i + duration] == current_regime:
                duration += 1

            # If duration is too short, replace with most common neighbor
            if duration < self.config.min_regime_duration:
                # Determine replacement regime
                if i == 0:
                    # Beginning of series, use next regime
                    replacement = filtered[i + duration] if i + duration < n else current_regime
                elif i + duration >= n:
                    # End of series, use previous regime
                    replacement = filtered[i - 1]
                else:
                    # Middle of series, use previous regime (more conservative)
                    replacement = filtered[i - 1]

                # Replace short segment
                filtered[i : i + duration] = replacement

                logger.debug(
                    "regime_duration_filtered",
                    position=i,
                    duration=duration,
                    from_regime=current_regime,
                    to_regime=replacement,
                )

            i += duration

        return filtered

    def fit(self, df: pl.DataFrame) -> None:
        """Train the regime detection model on historical data.

        This method:
        1. Computes features from price and volume data
        2. Trains a Gaussian HMM to identify regime patterns
        3. Maps HMM states to interpretable regime labels
        4. Stores the model for future predictions

        Args:
            df: Input DataFrame with historical price data.
                Must contain columns: timestamp, close
                Optional columns: volume, high, low

        Raises:
            ValueError: If DataFrame is empty, missing required columns,
                       or has insufficient data.
            RuntimeError: If HMM training fails.
        """
        if df.height == 0:
            raise ValueError("Cannot fit on empty DataFrame")

        if df.height < self.config.lookback_window:
            raise ValueError(
                f"Insufficient data for training: need at least {self.config.lookback_window} "
                f"rows, got {df.height}"
            )

        logger.info(
            "fitting_regime_detector",
            n_samples=df.height,
            n_regimes=self.config.n_regimes,
            lookback_window=self.config.lookback_window,
        )

        try:
            # Compute features
            df_features = self._compute_features(df)

            # Prepare feature matrix
            features = self._prepare_feature_matrix(df_features)

            # Standardize features
            features_scaled = self._standardize_features(features, fit=True)

            # Initialize and train HMM
            self._model = GaussianHMM(
                n_components=self.config.n_regimes,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
                random_state=self.config.random_state,
            )

            logger.debug("training_hmm", n_samples=len(features_scaled), n_features=features_scaled.shape[1])

            self._model.fit(features_scaled)

            # Predict states on training data to establish regime mapping
            states = self._model.predict(features_scaled)

            # Map states to regimes
            self._map_states_to_regimes(features, states)

            # Store training data reference (last N rows)
            self._training_data = df.tail(self.config.lookback_window)
            self._fitted = True

            logger.info(
                "regime_detector_fitted",
                converged=self._model.monitor_.converged,
                n_iter=self._model.monitor_.iter,
                final_score=self._model.score(features_scaled),
            )

        except Exception as e:
            logger.error("regime_detection_fitting_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to fit regime detector: {e}")

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict market regimes for the given data.

        Args:
            df: Input DataFrame with price data.
                Must contain the same columns as training data.

        Returns:
            DataFrame with original columns plus:
            - regime: Predicted regime label (BULL, BEAR, RANGE, CRISIS)
            - regime_probability: Probability of the predicted regime

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If DataFrame is invalid.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if df.height == 0:
            raise ValueError("Cannot predict on empty DataFrame")

        logger.info("predicting_regimes", n_samples=df.height)

        try:
            # Compute features
            df_features = self._compute_features(df)

            # Prepare feature matrix
            features = self._prepare_feature_matrix(df_features)

            # Standardize features
            features_scaled = self._standardize_features(features, fit=False)

            # Predict states
            states = self._model.predict(features_scaled)

            # Apply minimum duration filter
            states_filtered = self._apply_minimum_duration_filter(states)

            # Store predictions for get_current_regime and get_regime_probabilities
            self._last_predictions = states_filtered

            # Get state probabilities
            state_probs = self._model.predict_proba(features_scaled)

            # Map states to regimes
            regimes = np.array([self._regime_mapping[state].value for state in states_filtered])

            # Get regime probabilities (max probability for predicted state)
            regime_probs = np.array([state_probs[i, states_filtered[i]] for i in range(len(states_filtered))])

            # Create result DataFrame (need to match the cleaned feature DataFrame length)
            # Get the indices of non-null rows
            mask = df_features.select(self._feature_columns).select(
                pl.all_horizontal(pl.all().is_not_null()).alias("valid")
            )["valid"]

            # Add predictions to the original DataFrame
            result = df_features.with_columns(
                pl.when(mask)
                .then(pl.lit(None, dtype=pl.Utf8).alias("regime"))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
                .alias("regime")
            ).with_columns(
                pl.when(mask)
                .then(pl.lit(None, dtype=pl.Float64).alias("regime_probability"))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("regime_probability")
            )

            # Fill in the predictions where we have valid features
            valid_indices = mask.to_list()
            pred_idx = 0
            regime_values = []
            prob_values = []

            for is_valid in valid_indices:
                if is_valid:
                    regime_values.append(regimes[pred_idx])
                    prob_values.append(regime_probs[pred_idx])
                    pred_idx += 1
                else:
                    regime_values.append(None)
                    prob_values.append(None)

            result = result.with_columns(
                [
                    pl.Series("regime", regime_values),
                    pl.Series("regime_probability", prob_values),
                ]
            )

            # Drop intermediate feature columns
            result = result.drop(self._feature_columns)

            logger.info(
                "regime_prediction_complete",
                n_predictions=len(regimes),
                regime_distribution={
                    regime: int(np.sum(regimes == regime)) for regime in np.unique(regimes)
                },
            )

            return result

        except Exception as e:
            logger.error("regime_prediction_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to predict regimes: {e}")

    def get_current_regime(self) -> Regime:
        """Get the most recent predicted regime.

        Returns:
            The current market regime.

        Raises:
            RuntimeError: If model has not been fitted or predict() has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting current regime. Call fit() first.")

        if self._last_predictions is None or len(self._last_predictions) == 0:
            raise RuntimeError("No predictions available. Call predict() first.")

        last_state = self._last_predictions[-1]
        current_regime = self._regime_mapping[last_state]

        logger.debug("current_regime_retrieved", regime=current_regime.value)

        return current_regime

    def get_regime_probabilities(self) -> dict[Regime, float]:
        """Get probability distribution over all regimes for the current period.

        Returns:
            Dictionary mapping each regime to its probability.

        Raises:
            RuntimeError: If model has not been fitted or predict() has not been called.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError(
                "Model must be fitted before getting regime probabilities. Call fit() first."
            )

        if self._last_predictions is None or len(self._last_predictions) == 0:
            raise RuntimeError("No predictions available. Call predict() first.")

        # Get probability distribution from the model
        # Use the steady-state distribution
        # Compute steady state from transition matrix
        transmat = self._model.transmat_

        # Compute steady-state distribution (eigenvector with eigenvalue 1)
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary_dist = np.real(eigenvectors[:, stationary_idx])
        stationary_dist = stationary_dist / np.sum(stationary_dist)

        # Map states to regimes and aggregate probabilities
        regime_probs: dict[Regime, float] = {regime: 0.0 for regime in Regime}

        for state in range(self.config.n_regimes):
            regime = self._regime_mapping[state]
            regime_probs[regime] = regime_probs[regime] + float(stationary_dist[state])

        logger.debug("regime_probabilities_retrieved", probabilities=regime_probs)

        return regime_probs

    def get_transition_matrix(self) -> pl.DataFrame:
        """Get the regime transition probability matrix.

        Returns:
            DataFrame with rows and columns labeled by regime names,
            showing transition probabilities between regimes.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before getting transition matrix. Call fit() first.")

        # Get HMM transition matrix
        transmat = self._model.transmat_

        # Create regime-to-regime transition matrix
        regime_list = list(Regime)
        n_regimes_enum = len(regime_list)

        # Initialize regime transition matrix
        regime_transmat = np.zeros((n_regimes_enum, n_regimes_enum))

        # Aggregate state transitions into regime transitions
        for from_state in range(self.config.n_regimes):
            from_regime = self._regime_mapping[from_state]
            from_idx = regime_list.index(from_regime)

            for to_state in range(self.config.n_regimes):
                to_regime = self._regime_mapping[to_state]
                to_idx = regime_list.index(to_regime)

                regime_transmat[from_idx, to_idx] += transmat[from_state, to_state]

        # Normalize rows to sum to 1
        row_sums = regime_transmat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        regime_transmat = regime_transmat / row_sums

        # Create DataFrame
        regime_names = [regime.value for regime in regime_list]

        result = pl.DataFrame(
            regime_transmat,
            schema=dict.fromkeys(regime_names, pl.Float64),
        ).with_columns(pl.Series("from_regime", regime_names))

        # Reorder columns
        result = result.select(["from_regime"] + regime_names)

        logger.debug("transition_matrix_retrieved")

        return result

    def save_model(self, path: str | Path) -> None:
        """Save the fitted model to disk.

        Args:
            path: File path to save the model.

        Raises:
            RuntimeError: If model has not been fitted.
            OSError: If file cannot be written.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before saving. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("saving_regime_model", path=str(path))

        try:
            model_data = {
                "config": self.config,
                "model": self._model,
                "feature_columns": self._feature_columns,
                "regime_mapping": self._regime_mapping,
                "scaler_mean": self._scaler_mean,
                "scaler_std": self._scaler_std,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info("regime_model_saved", path=str(path))

        except Exception as e:
            logger.error("regime_model_save_failed", error=str(e), path=str(path), exc_info=True)
            raise OSError(f"Failed to save model to {path}: {e}")

    def load_model(self, path: str | Path) -> None:
        """Load a fitted model from disk.

        Args:
            path: File path to load the model from.

        Raises:
            OSError: If file cannot be read.
            ValueError: If loaded data is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise OSError(f"Model file not found: {path}")

        logger.info("loading_regime_model", path=str(path))

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.config = model_data["config"]
            self._model = model_data["model"]
            self._feature_columns = model_data["feature_columns"]
            self._regime_mapping = model_data["regime_mapping"]
            self._scaler_mean = model_data["scaler_mean"]
            self._scaler_std = model_data["scaler_std"]
            self._fitted = True

            logger.info("regime_model_loaded", path=str(path))

        except Exception as e:
            logger.error("regime_model_load_failed", error=str(e), path=str(path), exc_info=True)
            raise OSError(f"Failed to load model from {path}: {e}")

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been trained.

        Returns:
            True if model is fitted and ready for prediction.
        """
        return self._fitted and self._model is not None
