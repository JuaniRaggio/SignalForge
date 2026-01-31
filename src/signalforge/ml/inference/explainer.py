"""SHAP-based model explainability for SignalForge.

This module provides comprehensive explainability for machine learning models
using SHAP (SHapley Additive exPlanations) values. It enables understanding
which features contribute to predictions and by how much.

The module supports:
- Multiple SHAP methods (Kernel, Tree, Linear, Deep)
- Batch explanation processing
- Human-readable financial context explanations
- Visualization data generation for waterfall and summary plots
- Fallback to permutation importance if SHAP is unavailable

Key Classes:
    FeatureImportance: Individual feature contribution
    ExplanationResult: Complete explanation for a single prediction
    ExplainerConfig: Configuration for explainer behavior
    BaseExplainer: Abstract interface for all explainers
    ModelExplainer: Concrete SHAP-based implementation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureImportance:
    """Feature contribution to a prediction.

    Represents a single feature's impact on a model prediction,
    including both magnitude and direction of contribution.

    Attributes:
        feature: Name of the feature
        importance: Absolute SHAP value indicating contribution magnitude
        direction: Whether the feature pushed prediction positive or negative

    Examples:
        >>> feature = FeatureImportance(
        ...     feature="rsi_14",
        ...     importance=2.3,
        ...     direction="positive"
        ... )
    """

    feature: str
    importance: float
    direction: Literal["positive", "negative"]


@dataclass
class ExplanationResult:
    """Complete explanation for a single prediction.

    Contains all information needed to understand why a model
    made a specific prediction, including feature contributions
    and human-readable text.

    Attributes:
        prediction: The model's predicted value
        base_value: Expected value (model's average prediction)
        feature_contributions: List of all feature contributions
        top_features: Names of most important features
        summary_text: Human-readable explanation

    Examples:
        >>> result = ExplanationResult(
        ...     prediction=0.65,
        ...     base_value=0.50,
        ...     feature_contributions=[
        ...         FeatureImportance("rsi_14", 0.08, "positive"),
        ...         FeatureImportance("macd", 0.05, "positive"),
        ...     ],
        ...     top_features=["rsi_14", "macd"],
        ...     summary_text="Model predicts bullish signal..."
        ... )
    """

    prediction: float
    base_value: float
    feature_contributions: list[FeatureImportance]
    top_features: list[str]
    summary_text: str


@dataclass
class ExplainerConfig:
    """Configuration for model explainer.

    Controls the behavior of the explainer including which SHAP
    method to use and how many features to highlight.

    Attributes:
        method: SHAP method to use. Options:
            - "kernel": Model-agnostic KernelSHAP (default)
            - "tree": Fast TreeSHAP for tree-based models
            - "linear": LinearSHAP for linear models
            - "deep": DeepSHAP for neural networks
        n_samples: Number of background samples for KernelSHAP
        max_features: Maximum number of top features to show

    Examples:
        >>> config = ExplainerConfig(
        ...     method="tree",
        ...     n_samples=100,
        ...     max_features=10
        ... )
    """

    method: Literal["kernel", "tree", "linear", "deep"] = "kernel"
    n_samples: int = 100
    max_features: int = 10


class BaseExplainer(ABC):
    """Abstract base class for model explainers.

    Defines the interface that all explainer implementations must follow.
    This ensures consistency across different explainer types and provides
    a clear contract for explanation generation.

    All concrete explainer implementations must inherit from this class
    and implement all abstract methods.
    """

    @abstractmethod
    def explain(self, model: Any, X: pl.DataFrame) -> ExplanationResult:
        """Generate explanation for a single prediction.

        Args:
            model: Trained model to explain (must have predict method)
            X: Single row DataFrame with features

        Returns:
            ExplanationResult containing feature contributions and summary

        Raises:
            ValueError: If X contains multiple rows or invalid features
            RuntimeError: If explanation generation fails
        """
        pass

    @abstractmethod
    def explain_batch(self, model: Any, X: pl.DataFrame) -> list[ExplanationResult]:
        """Generate explanations for multiple predictions.

        Args:
            model: Trained model to explain (must have predict method)
            X: DataFrame with multiple rows of features

        Returns:
            List of ExplanationResult, one per input row

        Raises:
            ValueError: If X is empty or has invalid features
            RuntimeError: If explanation generation fails
        """
        pass

    @abstractmethod
    def get_feature_importance(self, model: Any, X: pl.DataFrame) -> list[FeatureImportance]:
        """Calculate global feature importance across samples.

        Args:
            model: Trained model to analyze
            X: DataFrame with features to analyze

        Returns:
            List of FeatureImportance sorted by importance (descending)

        Raises:
            ValueError: If X is empty or has invalid features
            RuntimeError: If importance calculation fails
        """
        pass


class ModelExplainer(BaseExplainer):
    """SHAP-based model explainer for SignalForge.

    Concrete implementation of BaseExplainer using SHAP values
    to explain model predictions. Supports multiple SHAP methods
    and falls back to permutation importance if SHAP is unavailable.

    The explainer can:
    - Explain individual predictions
    - Process batches of predictions
    - Calculate global feature importance
    - Generate human-readable explanations
    - Create visualization data

    Attributes:
        config: Explainer configuration
        _background_data: Background dataset for SHAP
        _explainer: Initialized SHAP explainer (lazy loaded)

    Examples:
        >>> import polars as pl
        >>> from signalforge.ml.inference.explainer import ModelExplainer, ExplainerConfig
        >>>
        >>> config = ExplainerConfig(method="tree", n_samples=100)
        >>> explainer = ModelExplainer(config)
        >>>
        >>> # Train your model
        >>> model = train_model(data)
        >>>
        >>> # Explain a prediction
        >>> X = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
        >>> explanation = explainer.explain(model, X)
        >>> print(explanation.summary_text)
    """

    def __init__(self, config: ExplainerConfig | None = None) -> None:
        """Initialize model explainer.

        Args:
            config: Explainer configuration. If None, uses defaults.
        """
        self.config = config or ExplainerConfig()
        self._background_data: pl.DataFrame | None = None
        self._explainer: Any = None

        logger.debug(
            "model_explainer_initialized",
            method=self.config.method,
            n_samples=self.config.n_samples,
            max_features=self.config.max_features,
        )

    def _initialize_explainer(self, model: Any, X: pl.DataFrame) -> None:
        """Initialize SHAP explainer with background data.

        Args:
            model: Model to explain
            X: Data to use for background samples

        Raises:
            ImportError: If SHAP library is not available
            RuntimeError: If explainer initialization fails
        """
        try:
            import shap
        except ImportError:
            logger.error("shap_import_failed")
            raise ImportError(
                "SHAP library is required for model explanation. "
                "Install with: pip install shap>=0.42.0"
            )

        try:
            # Sample background data if needed
            if X.height > self.config.n_samples:
                background_indices = np.random.choice(
                    X.height, size=self.config.n_samples, replace=False
                )
                self._background_data = X[background_indices]
            else:
                self._background_data = X

            logger.info(
                "initializing_shap_explainer",
                method=self.config.method,
                background_samples=self._background_data.height,
            )

            # Convert to numpy for SHAP
            background_array = self._background_data.to_numpy()

            # Initialize appropriate explainer based on method
            if self.config.method == "kernel":
                self._explainer = shap.KernelExplainer(
                    model.predict, background_array, link="identity"
                )
            elif self.config.method == "tree":
                # For tree-based models (XGBoost, LightGBM, RandomForest)
                self._explainer = shap.TreeExplainer(model)
            elif self.config.method == "linear":
                # For linear models
                self._explainer = shap.LinearExplainer(model, background_array)
            elif self.config.method == "deep":
                # For neural networks
                self._explainer = shap.DeepExplainer(model, background_array)
            else:
                raise ValueError(f"Unknown SHAP method: {self.config.method}")

            logger.info("shap_explainer_initialized", method=self.config.method)

        except Exception as e:
            logger.error(
                "explainer_initialization_failed",
                error=str(e),
                method=self.config.method,
                exc_info=True,
            )
            raise RuntimeError(f"Failed to initialize SHAP explainer: {e}")

    def _compute_shap_values(self, X: pl.DataFrame) -> np.ndarray:
        """Compute SHAP values for input data.

        Args:
            X: Input features

        Returns:
            Array of SHAP values (shape: [n_samples, n_features])

        Raises:
            RuntimeError: If SHAP value computation fails
        """
        try:
            X_array = X.to_numpy()

            if self.config.method == "kernel":
                # KernelExplainer returns Explanation object
                shap_values = self._explainer.shap_values(X_array)
            elif self.config.method == "tree":
                # TreeExplainer returns array directly
                explanation = self._explainer(X_array)
                shap_values = explanation.values
            elif self.config.method == "linear":
                # LinearExplainer returns array
                shap_values = self._explainer.shap_values(X_array)
            elif self.config.method == "deep":
                # DeepExplainer returns array
                shap_values = self._explainer.shap_values(X_array)
            else:
                raise ValueError(f"Unknown SHAP method: {self.config.method}")

            return np.array(shap_values)

        except Exception as e:
            logger.error(
                "shap_computation_failed",
                error=str(e),
                method=self.config.method,
                exc_info=True,
            )
            raise RuntimeError(f"Failed to compute SHAP values: {e}")

    def _fallback_permutation_importance(
        self, model: Any, X: pl.DataFrame
    ) -> list[FeatureImportance]:
        """Calculate feature importance using permutation method.

        Used as fallback when SHAP is unavailable or fails.

        Args:
            model: Model to analyze
            X: Input features

        Returns:
            List of FeatureImportance sorted by importance

        Raises:
            RuntimeError: If permutation importance calculation fails
        """
        logger.warning("using_permutation_importance_fallback")

        try:
            X_array = X.to_numpy()
            baseline_pred = model.predict(X_array)
            baseline_score = float(np.mean(baseline_pred))

            importances = []
            feature_names = X.columns

            for i, feature_name in enumerate(feature_names):
                # Permute feature
                X_permuted = X_array.copy()
                np.random.shuffle(X_permuted[:, i])

                # Calculate score change
                permuted_pred = model.predict(X_permuted)
                permuted_score = float(np.mean(permuted_pred))
                importance = abs(baseline_score - permuted_score)

                # Determine direction (simplified)
                direction: Literal["positive", "negative"] = (
                    "positive" if permuted_score < baseline_score else "negative"
                )

                importances.append(
                    FeatureImportance(
                        feature=feature_name, importance=importance, direction=direction
                    )
                )

            # Sort by importance (descending)
            importances.sort(key=lambda x: x.importance, reverse=True)

            logger.info("permutation_importance_calculated", n_features=len(importances))
            return importances

        except Exception as e:
            logger.error("permutation_importance_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to calculate permutation importance: {e}")

    def explain(self, model: Any, X: pl.DataFrame) -> ExplanationResult:
        """Generate explanation for a single prediction.

        Args:
            model: Trained model to explain
            X: Single row DataFrame with features

        Returns:
            ExplanationResult with feature contributions and summary

        Raises:
            ValueError: If X contains multiple rows
            RuntimeError: If explanation generation fails
        """
        if X.height != 1:
            raise ValueError(f"X must contain exactly one row, got {X.height}")

        if X.height == 0:
            raise ValueError("X cannot be empty")

        logger.info("generating_explanation", n_features=len(X.columns))

        try:
            # Initialize explainer if needed
            if self._explainer is None:
                self._initialize_explainer(model, X)

            # Get prediction
            X_array = X.to_numpy()
            prediction = float(model.predict(X_array)[0])

            # Compute SHAP values
            shap_values = self._compute_shap_values(X)

            # Get base value (expected value)
            if hasattr(self._explainer, "expected_value"):
                base_value = float(self._explainer.expected_value)
            else:
                # Fallback to mean prediction on background data
                assert self._background_data is not None
                bg_array = self._background_data.to_numpy()
                base_value = float(np.mean(model.predict(bg_array)))

            # Extract feature contributions for single sample
            feature_names = X.columns
            shap_values_single = shap_values[0] if shap_values.ndim > 1 else shap_values

            # Create feature importance list
            feature_contributions = []
            for i, feature_name in enumerate(feature_names):
                shap_val = float(shap_values_single[i])
                direction: Literal["positive", "negative"] = (
                    "positive" if shap_val >= 0 else "negative"
                )

                feature_contributions.append(
                    FeatureImportance(
                        feature=feature_name, importance=abs(shap_val), direction=direction
                    )
                )

            # Sort by importance
            feature_contributions.sort(key=lambda x: x.importance, reverse=True)

            # Get top features
            top_features = [fc.feature for fc in feature_contributions[: self.config.max_features]]

            # Generate summary text
            summary_text = generate_explanation_text(
                ExplanationResult(
                    prediction=prediction,
                    base_value=base_value,
                    feature_contributions=feature_contributions,
                    top_features=top_features,
                    summary_text="",  # Placeholder
                )
            )

            result = ExplanationResult(
                prediction=prediction,
                base_value=base_value,
                feature_contributions=feature_contributions,
                top_features=top_features,
                summary_text=summary_text,
            )

            logger.info(
                "explanation_generated",
                prediction=prediction,
                base_value=base_value,
                n_contributions=len(feature_contributions),
            )

            return result

        except Exception as e:
            logger.error("explanation_generation_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to generate explanation: {e}")

    def explain_batch(self, model: Any, X: pl.DataFrame) -> list[ExplanationResult]:
        """Generate explanations for multiple predictions.

        Args:
            model: Trained model to explain
            X: DataFrame with multiple rows of features

        Returns:
            List of ExplanationResult, one per input row

        Raises:
            ValueError: If X is empty
            RuntimeError: If explanation generation fails
        """
        if X.height == 0:
            raise ValueError("X cannot be empty")

        logger.info("generating_batch_explanations", n_samples=X.height)

        try:
            # Initialize explainer if needed
            if self._explainer is None:
                self._initialize_explainer(model, X)

            # Get predictions
            X_array = X.to_numpy()
            predictions = model.predict(X_array)

            # Compute SHAP values for all samples
            shap_values = self._compute_shap_values(X)

            # Get base value
            if hasattr(self._explainer, "expected_value"):
                base_value = float(self._explainer.expected_value)
            else:
                assert self._background_data is not None
                bg_array = self._background_data.to_numpy()
                base_value = float(np.mean(model.predict(bg_array)))

            # Generate explanation for each sample
            results = []
            feature_names = X.columns

            for idx in range(X.height):
                # Extract SHAP values for this sample
                shap_values_single = shap_values[idx]

                # Create feature contributions
                feature_contributions = []
                for i, feature_name in enumerate(feature_names):
                    shap_val = float(shap_values_single[i])
                    direction: Literal["positive", "negative"] = (
                        "positive" if shap_val >= 0 else "negative"
                    )

                    feature_contributions.append(
                        FeatureImportance(
                            feature=feature_name,
                            importance=abs(shap_val),
                            direction=direction,
                        )
                    )

                # Sort by importance
                feature_contributions.sort(key=lambda x: x.importance, reverse=True)

                # Get top features
                top_features = [
                    fc.feature for fc in feature_contributions[: self.config.max_features]
                ]

                # Create result
                result = ExplanationResult(
                    prediction=float(predictions[idx]),
                    base_value=base_value,
                    feature_contributions=feature_contributions,
                    top_features=top_features,
                    summary_text="",  # Generated later if needed
                )

                # Generate summary text
                result.summary_text = generate_explanation_text(result)

                results.append(result)

            logger.info("batch_explanations_generated", n_explanations=len(results))
            return results

        except Exception as e:
            logger.error("batch_explanation_generation_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to generate batch explanations: {e}")

    def get_feature_importance(self, model: Any, X: pl.DataFrame) -> list[FeatureImportance]:
        """Calculate global feature importance across samples.

        Args:
            model: Trained model to analyze
            X: DataFrame with features to analyze

        Returns:
            List of FeatureImportance sorted by importance (descending)

        Raises:
            ValueError: If X is empty
            RuntimeError: If importance calculation fails
        """
        if X.height == 0:
            raise ValueError("X cannot be empty")

        logger.info("calculating_global_feature_importance", n_samples=X.height)

        try:
            # Try SHAP-based importance
            try:
                # Initialize explainer if needed
                if self._explainer is None:
                    self._initialize_explainer(model, X)

                # Compute SHAP values
                shap_values = self._compute_shap_values(X)

                # Calculate mean absolute SHAP values per feature
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

                # Calculate mean SHAP values to determine direction
                mean_shap = np.mean(shap_values, axis=0)

                feature_names = X.columns
                importances = []

                for i, feature_name in enumerate(feature_names):
                    direction: Literal["positive", "negative"] = (
                        "positive" if mean_shap[i] >= 0 else "negative"
                    )
                    importances.append(
                        FeatureImportance(
                            feature=feature_name,
                            importance=float(mean_abs_shap[i]),
                            direction=direction,
                        )
                    )

                # Sort by importance
                importances.sort(key=lambda x: x.importance, reverse=True)

                logger.info(
                    "global_feature_importance_calculated",
                    n_features=len(importances),
                    top_feature=importances[0].feature if importances else None,
                )

                return importances

            except Exception:
                # Fallback to permutation importance
                logger.warning("shap_failed_using_fallback")
                return self._fallback_permutation_importance(model, X)

        except Exception as e:
            logger.error("feature_importance_calculation_failed", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to calculate feature importance: {e}")


def generate_explanation_text(result: ExplanationResult) -> str:
    """Generate human-readable explanation for prediction.

    Creates a natural language summary of the prediction with
    financial context about feature contributions.

    Args:
        result: ExplanationResult to explain

    Returns:
        Human-readable explanation text

    Examples:
        >>> result = ExplanationResult(...)
        >>> text = generate_explanation_text(result)
        >>> print(text)
        The model predicts a value of 0.65 (base: 0.50).
        Top contributing features:
        - rsi_14 (POSITIVE +0.08): RSI indicates overbought conditions
        - macd (POSITIVE +0.05): MACD shows bullish momentum
    """
    # Determine prediction sentiment
    prediction_diff = result.prediction - result.base_value
    if prediction_diff > 0.1:
        sentiment = "strongly positive"
    elif prediction_diff > 0.01:
        sentiment = "slightly positive"
    elif prediction_diff < -0.1:
        sentiment = "strongly negative"
    elif prediction_diff < -0.01:
        sentiment = "slightly negative"
    else:
        sentiment = "neutral"

    # Build explanation
    lines = [
        f"The model predicts a value of {result.prediction:.4f} "
        f"(base: {result.base_value:.4f}, {sentiment} deviation)."
    ]

    if result.feature_contributions:
        lines.append("\nTop contributing features:")

        # Show top features with financial context
        for contrib in result.feature_contributions[: min(5, len(result.feature_contributions))]:
            direction_symbol = "+" if contrib.direction == "positive" else "-"
            direction_text = contrib.direction.upper()

            # Add financial context for known indicators
            context = _get_financial_context(contrib.feature, contrib.importance, contrib.direction)

            lines.append(
                f"- {contrib.feature} ({direction_text} {direction_symbol}{contrib.importance:.4f}): "
                f"{context}"
            )

    return "\n".join(lines)


def _get_financial_context(feature: str, importance: float, direction: str) -> str:
    """Get financial context for a feature.

    Args:
        feature: Feature name
        importance: Contribution magnitude
        direction: Contribution direction

    Returns:
        Human-readable context string
    """
    feature_lower = feature.lower()

    # RSI indicators
    if "rsi" in feature_lower:
        if direction == "positive":
            return "RSI indicates overbought conditions, suggesting potential resistance"
        else:
            return "RSI indicates oversold conditions, suggesting potential support"

    # MACD indicators
    if "macd" in feature_lower:
        if direction == "positive":
            return "MACD shows bullish momentum with positive divergence"
        else:
            return "MACD shows bearish momentum with negative divergence"

    # Moving averages
    if "sma" in feature_lower or "ema" in feature_lower or "ma_" in feature_lower:
        if direction == "positive":
            return "Moving average indicates upward trend strength"
        else:
            return "Moving average indicates downward trend pressure"

    # Volume indicators
    if "volume" in feature_lower:
        if direction == "positive":
            return "High volume supports the directional move"
        else:
            return "Low volume suggests weak conviction"

    # Bollinger Bands
    if "bb" in feature_lower or "bollinger" in feature_lower:
        if direction == "positive":
            return "Price approaching upper band, indicating strength"
        else:
            return "Price approaching lower band, indicating weakness"

    # ATR / Volatility
    if "atr" in feature_lower or "volatility" in feature_lower:
        if direction == "positive":
            return "Increased volatility suggests larger potential moves"
        else:
            return "Decreased volatility suggests consolidation"

    # Default generic context
    magnitude = "strongly" if importance > 0.1 else "moderately"
    return f"This feature {magnitude} influences the prediction {direction}ly"


def plot_waterfall(explanation: ExplanationResult) -> dict[str, Any]:
    """Generate waterfall plot data for explanation.

    Creates data structure suitable for plotting a waterfall chart
    showing how features contribute to the prediction.

    Args:
        explanation: ExplanationResult to visualize

    Returns:
        Dictionary containing:
            - features: List of feature names
            - values: List of SHAP values
            - base_value: Expected value
            - prediction: Final prediction

    Examples:
        >>> explanation = explainer.explain(model, X)
        >>> data = plot_waterfall(explanation)
        >>> # Use data with plotting library of choice
    """
    features = []
    values = []

    for contrib in explanation.feature_contributions[
        : min(10, len(explanation.feature_contributions))
    ]:
        features.append(contrib.feature)
        # Use signed value for waterfall
        value = contrib.importance if contrib.direction == "positive" else -contrib.importance
        values.append(value)

    return {
        "features": features,
        "values": values,
        "base_value": explanation.base_value,
        "prediction": explanation.prediction,
    }


def plot_summary(explanations: list[ExplanationResult]) -> dict[str, Any]:
    """Generate summary plot data for multiple explanations.

    Creates data structure suitable for plotting a summary chart
    showing feature importance across multiple predictions.

    Args:
        explanations: List of ExplanationResults

    Returns:
        Dictionary containing:
            - features: List of unique feature names
            - importances: Mean absolute importance per feature
            - shap_values: 2D array of SHAP values [n_samples, n_features]

    Examples:
        >>> explanations = explainer.explain_batch(model, X)
        >>> data = plot_summary(explanations)
        >>> # Use data with plotting library of choice
    """
    if not explanations:
        return {"features": [], "importances": [], "shap_values": []}

    # Collect all unique features
    all_features = set()
    for exp in explanations:
        for contrib in exp.feature_contributions:
            all_features.add(contrib.feature)

    features = sorted(all_features)

    # Build SHAP values matrix
    shap_values = []
    for exp in explanations:
        row = []
        contrib_dict = {
            fc.feature: (fc.importance if fc.direction == "positive" else -fc.importance)
            for fc in exp.feature_contributions
        }
        for feature in features:
            row.append(contrib_dict.get(feature, 0.0))
        shap_values.append(row)

    shap_array = np.array(shap_values)

    # Calculate mean absolute importance per feature
    importances = np.mean(np.abs(shap_array), axis=0).tolist()

    return {
        "features": features,
        "importances": importances,
        "shap_values": shap_array.tolist(),
    }
