"""Abstract base class for all prediction models.

This module defines the interface that all prediction models must implement,
ensuring consistency across different model types (statistical, ML, DL).

The BasePredictor class enforces a contract for:
- Training models on historical data
- Generating predictions for future periods
- Evaluating model performance on test data
- Tracking model training state

All concrete model implementations must inherit from BasePredictor and
implement all abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class BasePredictor(ABC):
    """Abstract base class for all prediction models.

    This class defines the interface that all prediction models must implement.
    It ensures consistency across different model types and provides a clear
    contract for model training, prediction, and evaluation.

    Subclasses must implement all abstract methods and properties to be instantiable.

    Attributes:
        The specific attributes depend on the concrete implementation.

    Examples:
        Creating a custom predictor:

        >>> class CustomPredictor(BasePredictor):
        ...     def __init__(self):
        ...         self._fitted = False
        ...         self._model = None
        ...
        ...     def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        ...         # Training logic here
        ...         self._fitted = True
        ...
        ...     def predict(self, horizon: int) -> pl.DataFrame:
        ...         if not self.is_fitted:
        ...             raise RuntimeError("Model must be fitted before prediction")
        ...         # Prediction logic here
        ...         return pl.DataFrame()
        ...
        ...     def evaluate(self, test_df: pl.DataFrame) -> dict[str, float]:
        ...         # Evaluation logic here
        ...         return {"rmse": 0.0, "mae": 0.0}
        ...
        ...     @property
        ...     def is_fitted(self) -> bool:
        ...         return self._fitted
    """

    @abstractmethod
    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Train the model on historical data.

        This method trains the prediction model using the provided historical
        time series data. The model should learn patterns from the data to
        make future predictions.

        Args:
            df: Input DataFrame with historical time series data.
                Must contain at least a timestamp column and the target column.
            target_column: Name of the column to predict. Defaults to "close".

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
            RuntimeError: If model fitting fails.

        Note:
            After successful fitting, the is_fitted property should return True.
        """
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pl.DataFrame:
        """Generate predictions for future periods.

        This method generates predictions for the specified number of periods
        into the future. The model must be fitted before calling this method.

        Args:
            horizon: Number of periods to predict into the future.
                     Must be a positive integer.

        Returns:
            DataFrame containing predictions with at least two columns:
            - timestamp: The timestamp for each prediction period
            - prediction: The predicted value for each period

            Additional columns may include confidence intervals or other
            model-specific information.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If horizon is not positive.

        Note:
            The timestamp column should contain future timestamps starting
            from the last timestamp in the training data.
        """
        pass

    @abstractmethod
    def evaluate(self, test_df: pl.DataFrame) -> dict[str, float]:
        """Evaluate model performance on test data.

        This method evaluates the model's predictions against actual test data
        and returns a dictionary of evaluation metrics.

        Args:
            test_df: DataFrame with actual values for comparison.
                     Must contain the same columns as the training data.

        Returns:
            Dictionary mapping metric names to their values. At minimum,
            should include:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error
            - direction_accuracy: Percentage of predictions with correct direction

            Additional metrics may be included based on model type.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If test_df is empty or missing required columns.

        Note:
            The model should generate predictions for the test period and
            compare them against the actual values in test_df.
        """
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if model has been trained.

        Returns:
            True if the model has been successfully fitted and is ready
            for prediction, False otherwise.

        Note:
            This property should be checked before calling predict() or
            evaluate() methods to avoid runtime errors.
        """
        pass
