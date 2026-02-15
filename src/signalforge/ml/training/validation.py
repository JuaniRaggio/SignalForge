"""Walk-forward validation for time series models.

This module provides time series cross-validation with proper temporal ordering,
purging to prevent look-ahead bias, and embargo periods to prevent data leakage.

The WalkForwardValidator implements a rigorous validation scheme specifically
designed for financial time series that respects temporal dependencies.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import polars as pl

    from signalforge.ml.models.base import BasePredictor

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Results from walk-forward cross-validation.

    Attributes:
        metrics: Dictionary mapping metric names to lists of fold values.
                Example: {"mse": [0.1, 0.15, 0.12], "mae": [0.05, 0.06, 0.055]}
        mean_metrics: Dictionary of mean values for each metric across folds.
        std_metrics: Dictionary of standard deviation values for each metric.
        predictions: DataFrame containing all out-of-sample predictions across folds.
        fold_info: List of dictionaries with metadata for each fold (train/test dates).

    Examples:
        >>> result = ValidationResult(
        ...     metrics={"mse": [0.1, 0.15], "mae": [0.05, 0.06]},
        ...     mean_metrics={"mse": 0.125, "mae": 0.055},
        ...     std_metrics={"mse": 0.025, "mae": 0.005},
        ...     predictions=pl.DataFrame(),
        ...     fold_info=[{"fold": 1, "train_start": "2020-01-01", "train_end": "2020-12-31"}]
        ... )
    """

    metrics: dict[str, list[float]]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    predictions: pl.DataFrame
    fold_info: list[dict[str, Any]]


class WalkForwardValidator:
    """Walk-forward validation with purging and embargo for time series.

    This validator implements a rigorous cross-validation scheme for time series
    that prevents look-ahead bias and data leakage through:

    1. Temporal ordering: Training always precedes testing
    2. Purging: Removing observations from training set that overlap with test set
    3. Embargo: Excluding recent observations before test set to prevent leakage

    The validator divides the time series into multiple train/test splits, where
    each split advances forward in time (hence "walk-forward").

    Attributes:
        n_splits: Number of train/test splits to generate.
        train_size: Number of observations in each training set. If None, uses expanding window.
        test_size: Number of observations in each test set.
        purge_days: Number of days to purge from training set before test period.
        embargo_days: Number of days to exclude at the end of training set.

    Examples:
        Basic usage:

        >>> import polars as pl
        >>> from signalforge.ml.training.validation import WalkForwardValidator
        >>>
        >>> df = pl.DataFrame({
        ...     "timestamp": pl.date_range(start="2020-01-01", periods=100, interval="1d"),
        ...     "value": range(100)
        ... })
        >>>
        >>> validator = WalkForwardValidator(n_splits=5, test_size=10, purge_days=2)
        >>> for train_df, test_df in validator.split(df, date_column="timestamp"):
        ...     # Train and evaluate model on this fold
        ...     pass

        With expanding window (train_size=None):

        >>> validator = WalkForwardValidator(n_splits=3, train_size=None, test_size=20)
        >>> # Each fold will use all available historical data up to that point

        With fixed window (train_size=50):

        >>> validator = WalkForwardValidator(n_splits=3, train_size=50, test_size=20)
        >>> # Each fold will use exactly 50 observations for training
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int = 20,
        purge_days: int = 5,
        embargo_days: int = 2,
    ) -> None:
        """Initialize the walk-forward validator.

        Args:
            n_splits: Number of train/test splits to generate. Must be >= 2.
            train_size: Number of observations per training set. If None, uses
                       expanding window (all data up to test set).
            test_size: Number of observations in each test set. Must be > 0.
            purge_days: Days to remove from training set that overlap with test.
                       Prevents look-ahead bias. Must be >= 0.
            embargo_days: Days to exclude at end of training before test period.
                         Prevents information leakage. Must be >= 0.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if test_size <= 0:
            raise ValueError(f"test_size must be positive, got {test_size}")
        if train_size is not None and train_size <= 0:
            raise ValueError(f"train_size must be positive or None, got {train_size}")
        if purge_days < 0:
            raise ValueError(f"purge_days must be non-negative, got {purge_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be non-negative, got {embargo_days}")

        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_days = purge_days
        self.embargo_days = embargo_days

        logger.info(
            "Initialized WalkForwardValidator",
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )

    def split(
        self, df: pl.DataFrame, date_column: str = "timestamp"
    ) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """Generate train/test splits with purging and embargo.

        Yields successive train/test splits that walk forward through time.
        Each split maintains temporal ordering and applies purging/embargo rules.

        Args:
            df: Input DataFrame with time series data. Must contain date_column.
            date_column: Name of the column containing timestamps. Must be
                        datetime type and sorted in ascending order.

        Yields:
            Tuples of (train_df, test_df) for each fold.

        Raises:
            ValueError: If date_column is missing or data is insufficient for n_splits.
            RuntimeError: If date_column is not sorted in ascending order.

        Examples:
            >>> import polars as pl
            >>> df = pl.DataFrame({
            ...     "timestamp": pl.date_range(start="2020-01-01", periods=100, interval="1d"),
            ...     "price": range(100)
            ... })
            >>> validator = WalkForwardValidator(n_splits=3, test_size=10)
            >>> for i, (train, test) in enumerate(validator.split(df)):
            ...     print(f"Fold {i}: train size={len(train)}, test size={len(test)}")
        """
        import polars as pl

        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")

        # Ensure data is sorted by date
        df_sorted = df.sort(date_column)

        # Check if sorting changed the order (data was not already sorted)
        if not df_sorted[date_column].equals(df[date_column]):
            logger.warning(
                "DataFrame was not sorted by date column, automatically sorted",
                date_column=date_column,
            )

        n_samples = len(df_sorted)
        min_required = self.test_size * self.n_splits
        if self.train_size is not None:
            min_required += self.train_size
        else:
            min_required += self.test_size  # Need at least some training data

        if n_samples < min_required:
            raise ValueError(
                f"Insufficient data: {n_samples} samples available, "
                f"but need at least {min_required} for {self.n_splits} splits"
            )

        # Calculate split indices
        test_starts = self._calculate_test_starts(n_samples)

        for fold_idx, test_start in enumerate(test_starts):
            test_end = min(test_start + self.test_size, n_samples)

            # Calculate training set boundaries
            if self.train_size is None:
                # Expanding window: use all data from start to test_start
                train_start = 0
                train_end = test_start
            else:
                # Fixed window: use train_size observations before test_start
                train_start = max(0, test_start - self.train_size)
                train_end = test_start

            # Apply embargo: exclude last embargo_days from training
            if self.embargo_days > 0:
                train_end_date = df_sorted[date_column][train_end - 1]
                embargo_cutoff = train_end_date - timedelta(days=self.embargo_days)
                train_end = (
                    df_sorted.filter(pl.col(date_column) <= embargo_cutoff)
                    .select(pl.count())
                    .item()
                )

            # Apply purging: remove training samples that overlap with test period
            if self.purge_days > 0:
                test_start_date = df_sorted[date_column][test_start]
                purge_cutoff = test_start_date - timedelta(days=self.purge_days)
                # Keep only training data before purge cutoff
                train_df = df_sorted[train_start:train_end].filter(
                    pl.col(date_column) < purge_cutoff
                )
            else:
                train_df = df_sorted[train_start:train_end]

            test_df = df_sorted[test_start:test_end]

            if len(train_df) == 0:
                logger.warning(
                    "Empty training set in fold, skipping",
                    fold=fold_idx + 1,
                    train_start=train_start,
                    train_end=train_end,
                )
                continue

            if len(test_df) == 0:
                logger.warning(
                    "Empty test set in fold, skipping",
                    fold=fold_idx + 1,
                    test_start=test_start,
                    test_end=test_end,
                )
                continue

            logger.debug(
                "Generated fold",
                fold=fold_idx + 1,
                train_samples=len(train_df),
                test_samples=len(test_df),
                train_start_date=str(train_df[date_column][0]),
                train_end_date=str(train_df[date_column][-1]),
                test_start_date=str(test_df[date_column][0]),
                test_end_date=str(test_df[date_column][-1]),
            )

            yield train_df, test_df

    def _calculate_test_starts(self, n_samples: int) -> list[int]:
        """Calculate starting indices for test sets.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of starting indices for each test fold.
        """
        if self.train_size is None:
            # For expanding window, space test sets evenly through data
            # Reserve last portion for test sets
            total_test_size = self.test_size * self.n_splits
            available_for_tests = n_samples - self.test_size  # Need at least one train sample
            step = (available_for_tests - total_test_size) // (self.n_splits - 1)
            step = max(step, self.test_size)  # Ensure no overlap between test sets

            test_starts = []
            for i in range(self.n_splits):
                start = self.test_size + i * step  # Start after minimum training data
                if start + self.test_size <= n_samples:
                    test_starts.append(start)
        else:
            # For fixed window, test sets follow each other with train_size separation
            first_test_start = self.train_size
            step = self.test_size

            test_starts = []
            for i in range(self.n_splits):
                start = first_test_start + i * step
                if start + self.test_size <= n_samples:
                    test_starts.append(start)

        return test_starts

    def cross_validate(
        self,
        model: BasePredictor,
        X: pl.DataFrame,
        y: pl.Series,
        metrics: list[str] | None = None,
    ) -> ValidationResult:
        """Run walk-forward cross-validation on a model.

        Performs complete cross-validation by training and evaluating the model
        on each fold. Calculates specified metrics and aggregates results.

        Args:
            model: Model instance to validate. Must implement BasePredictor interface.
            X: Feature matrix with all data. Must contain timestamp column.
            y: Target series corresponding to X.
            metrics: List of metric names to calculate. Options: "mse", "mae",
                    "rmse", "directional_accuracy". If None, uses default set.

        Returns:
            ValidationResult containing metrics, predictions, and fold information.

        Raises:
            ValueError: If X and y have incompatible shapes or metrics are invalid.
            RuntimeError: If model training or prediction fails.

        Examples:
            >>> import polars as pl
            >>> from signalforge.ml.training.validation import WalkForwardValidator
            >>>
            >>> X = pl.DataFrame({
            ...     "timestamp": pl.date_range(start="2020-01-01", periods=100, interval="1d"),
            ...     "feature1": range(100),
            ...     "feature2": range(100, 200)
            ... })
            >>> y = pl.Series([float(i) for i in range(100)])
            >>>
            >>> validator = WalkForwardValidator(n_splits=3, test_size=10)
            >>> # Assume model is a concrete implementation of BasePredictor
            >>> # result = validator.cross_validate(model, X, y, metrics=["mse", "mae"])
        """
        import numpy as np
        import polars as pl

        if metrics is None:
            metrics = ["mse", "mae", "directional_accuracy"]

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: X={len(X)}, y={len(y)}")

        # Combine X and y for splitting
        df = X.with_columns(pl.Series("__target__", y))

        # Storage for results
        fold_metrics: dict[str, list[float]] = {metric: [] for metric in metrics}
        all_predictions = []
        fold_info_list = []

        for fold_idx, (train_df, test_df) in enumerate(self.split(df, date_column="timestamp")):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_splits}")

            # Split features and target
            # Drop timestamp and target from features
            X_train = train_df.drop(["__target__", "timestamp"])
            y_train = train_df["__target__"]
            X_test = test_df.drop(["__target__", "timestamp"])
            y_test = test_df["__target__"]

            # Train model
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Model training failed on fold {fold_idx + 1}", error=str(e))
                raise RuntimeError(f"Model training failed on fold {fold_idx + 1}: {e}") from e

            # Generate predictions
            try:
                predictions = model.predict(X_test)
                y_pred = np.array([p.prediction for p in predictions])
            except Exception as e:
                logger.error(f"Model prediction failed on fold {fold_idx + 1}", error=str(e))
                raise RuntimeError(f"Model prediction failed on fold {fold_idx + 1}: {e}") from e

            # Calculate metrics
            # Note: For sequence models (LSTM), predictions may be fewer than test samples
            # Align by taking the last N predictions where N is len(y_pred)
            y_test_np = y_test.to_numpy()

            # Align predictions with targets
            if len(y_pred) < len(y_test_np):
                # Take last N samples from y_test to align with predictions
                y_test_np = y_test_np[-len(y_pred):]
            elif len(y_pred) > len(y_test_np):
                # Take last N predictions to align with y_test
                y_pred = y_pred[-len(y_test_np):]

            fold_result = self._calculate_metrics(y_test_np, y_pred, metrics)

            for metric in metrics:
                fold_metrics[metric].append(fold_result[metric])

            # Store predictions
            # Use the aligned lengths for creating the DataFrame
            n_preds = len(y_pred)
            test_timestamps = test_df.select("timestamp").tail(n_preds)

            pred_df = test_timestamps.with_columns(
                [
                    pl.Series("actual", y_test_np),
                    pl.Series("predicted", y_pred),
                    pl.lit(fold_idx + 1).alias("fold"),
                ]
            )
            all_predictions.append(pred_df)

            # Store fold info
            fold_info_list.append(
                {
                    "fold": fold_idx + 1,
                    "train_start": str(train_df["timestamp"][0]),
                    "train_end": str(train_df["timestamp"][-1]),
                    "test_start": str(test_df["timestamp"][0]),
                    "test_end": str(test_df["timestamp"][-1]),
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    **fold_result,
                }
            )

        # Aggregate results
        mean_metrics = {metric: float(np.mean(values)) for metric, values in fold_metrics.items()}
        std_metrics = {metric: float(np.std(values)) for metric, values in fold_metrics.items()}

        predictions_df = pl.concat(all_predictions) if all_predictions else pl.DataFrame()

        logger.info(
            "Cross-validation complete",
            n_folds=self.n_splits,
            mean_metrics=mean_metrics,
        )

        return ValidationResult(
            metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            predictions=predictions_df,
            fold_info=fold_info_list,
        )

    def _calculate_metrics(
        self, y_true: Any, y_pred: Any, metrics: list[str]
    ) -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            metrics: List of metric names to calculate.

        Returns:
            Dictionary mapping metric names to their values.

        Raises:
            ValueError: If an unknown metric name is provided.
        """
        import numpy as np

        result = {}

        for metric in metrics:
            if metric == "mse":
                result[metric] = float(np.mean((y_true - y_pred) ** 2))
            elif metric == "mae":
                result[metric] = float(np.mean(np.abs(y_true - y_pred)))
            elif metric == "rmse":
                result[metric] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            elif metric == "directional_accuracy":
                # Calculate percentage of correct directional predictions
                if len(y_true) > 1:
                    true_direction = np.sign(np.diff(y_true))
                    pred_direction = np.sign(np.diff(y_pred))
                    result[metric] = float(np.mean(true_direction == pred_direction))
                else:
                    result[metric] = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return result
