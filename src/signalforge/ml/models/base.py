"""Abstract base class for all prediction models.

This module defines the interface that all prediction models must implement,
ensuring consistency across different model types (statistical, ML, DL).

The BasePredictor class enforces a contract for:
- Training models on historical data
- Generating predictions for future periods
- Saving and loading models from disk
- Feature importance extraction
- Optional ONNX export for production deployment

All concrete model implementations must inherit from BasePredictor and
implement all abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl


@dataclass
class PredictionResult:
    """Container for model prediction results.

    Attributes:
        symbol: The ticker symbol being predicted
        timestamp: The timestamp of the prediction
        horizon_days: Number of days ahead this prediction is for
        prediction: The predicted value
        confidence: Confidence score for this prediction (0.0-1.0)
        lower_bound: Optional lower bound of prediction interval
        upper_bound: Optional upper bound of prediction interval
        model_name: Name of the model that generated this prediction
        model_version: Version string of the model

    Examples:
        >>> from datetime import datetime
        >>> result = PredictionResult(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     horizon_days=1,
        ...     prediction=150.25,
        ...     confidence=0.85,
        ...     lower_bound=148.0,
        ...     upper_bound=152.5,
        ...     model_name="LSTM",
        ...     model_version="1.0.0"
        ... )
    """

    symbol: str
    timestamp: datetime
    horizon_days: int
    prediction: float
    confidence: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    model_name: str = ""
    model_version: str = ""


class BasePredictor(ABC):
    """Abstract base class for all prediction models.

    This class defines the interface that all prediction models must implement.
    It ensures consistency across different model types and provides a clear
    contract for model training, prediction, evaluation, and persistence.

    Subclasses must implement all abstract methods to be instantiable.

    Attributes:
        model_name: Human-readable name of the model
        model_version: Version string for the model

    Examples:
        Creating a custom predictor:

        >>> import polars as pl
        >>> from signalforge.ml.models.base import BasePredictor, PredictionResult
        >>>
        >>> class CustomPredictor(BasePredictor):
        ...     model_name = "CustomModel"
        ...     model_version = "1.0.0"
        ...
        ...     def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs) -> "CustomPredictor":
        ...         # Training logic here
        ...         return self
        ...
        ...     def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        ...         # Prediction logic here
        ...         return []
        ...
        ...     def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        ...         # Probability prediction logic here
        ...         return pl.DataFrame()
    """

    model_name: str
    model_version: str

    @abstractmethod
    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> BasePredictor:
        """Train the model on historical data.

        This method trains the prediction model using the provided feature matrix
        and target series. The model should learn patterns from the data to make
        future predictions.

        Args:
            X: Feature matrix with historical time series data.
               Each row represents a time point, columns are features.
            y: Target series containing the values to predict.
            **kwargs: Additional model-specific training parameters.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X and y have incompatible shapes or contain invalid data.
            RuntimeError: If model fitting fails.

        Note:
            After successful fitting, the model should be ready for prediction.

        Examples:
            >>> import polars as pl
            >>> X = pl.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]})
            >>> y = pl.Series([10.0, 20.0, 30.0])
            >>> model = CustomPredictor()
            >>> model.fit(X, y)
        """
        ...

    @abstractmethod
    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Generate predictions for the given input features.

        This method generates predictions for the provided feature matrix.
        The model must be fitted before calling this method.

        Args:
            X: Feature matrix for which to generate predictions.
               Must have the same features as the training data.

        Returns:
            List of PredictionResult objects, one per row in X.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has incompatible features.

        Examples:
            >>> import polars as pl
            >>> X_test = pl.DataFrame({"feature1": [4.0], "feature2": [7.0]})
            >>> predictions = model.predict(X_test)
            >>> print(predictions[0].prediction)
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return prediction probabilities or confidence scores.

        This method returns probability distributions or confidence scores
        for predictions. The exact format depends on the model type.

        For regression models, this might return confidence intervals.
        For classification models, this returns class probabilities.

        Args:
            X: Feature matrix for which to generate probability predictions.

        Returns:
            DataFrame containing probability/confidence information.
            Structure depends on the specific model implementation.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has incompatible features.

        Examples:
            >>> import polars as pl
            >>> X_test = pl.DataFrame({"feature1": [4.0], "feature2": [7.0]})
            >>> probas = model.predict_proba(X_test)
        """
        ...

    def save(self, path: str) -> None:
        """Save model to disk.

        Serializes the model and saves it to the specified path.
        The default implementation is not provided and should be
        overridden by subclasses with appropriate serialization logic.

        Args:
            path: File path where the model should be saved.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            IOError: If saving fails due to file system errors.

        Examples:
            >>> model.save("/path/to/model.pkl")
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    @classmethod
    def load(cls, path: str) -> BasePredictor:
        """Load model from disk.

        Deserializes and loads a model from the specified path.
        The default implementation is not provided and should be
        overridden by subclasses with appropriate deserialization logic.

        Args:
            path: File path from which to load the model.

        Returns:
            Loaded model instance.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            IOError: If loading fails due to file system errors.
            ValueError: If the file contains invalid model data.

        Examples:
            >>> model = CustomPredictor.load("/path/to/model.pkl")
        """
        raise NotImplementedError(f"{cls.__name__} does not implement load()")

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances if available.

        Returns a dictionary mapping feature names to their importance scores.
        The importance metric depends on the model type (e.g., Gini importance
        for tree models, coefficient magnitude for linear models).

        Returns:
            Dictionary mapping feature names to importance scores.
            Returns empty dict if the model does not support feature importance.

        Examples:
            >>> importances = model.get_feature_importance()
            >>> for feature, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            ...     print(f"{feature}: {score:.4f}")
        """
        return {}

    def to_onnx(self, path: str) -> None:
        """Export model to ONNX format for production deployment.

        Converts the model to ONNX format and saves it to the specified path.
        ONNX provides a standardized format for model interoperability across
        different frameworks and platforms.

        Args:
            path: File path where the ONNX model should be saved.

        Raises:
            NotImplementedError: If the model does not support ONNX export.
            RuntimeError: If ONNX conversion fails.

        Note:
            Not all models support ONNX export. This is an optional feature.

        Examples:
            >>> model.to_onnx("/path/to/model.onnx")
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support ONNX export")
