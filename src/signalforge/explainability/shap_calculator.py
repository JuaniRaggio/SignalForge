"""SHAP value calculator for model explanations.

This module provides SHAP calculation utilities that can be used by the
ExplanationService to generate SHAP values for predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import shap
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


class SHAPCalculator:
    """Calculate SHAP values for model explanations.

    This class provides a simplified interface for calculating SHAP values
    from trained models, handling different model types automatically.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: pl.DataFrame | NDArray[np.floating[Any]] | None = None,
    ) -> None:
        """Initialize SHAP calculator.

        Args:
            model: Trained model (sklearn, torch, etc.)
            feature_names: List of feature names in prediction order
            background_data: Background dataset for SHAP explainer (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer: shap.Explainer | None = None
        self.explainer_type: str | None = None

        logger.info(
            "shap_calculator_initialized",
            num_features=len(feature_names),
            has_background_data=background_data is not None,
        )

    def _prepare_background_data(
        self,
    ) -> NDArray[np.floating[Any]] | None:
        """Prepare background data for SHAP explainer.

        Returns:
            Numpy array of background data or None
        """
        if self.background_data is None:
            return None

        if isinstance(self.background_data, pl.DataFrame):
            return self.background_data.select(self.feature_names).to_numpy()

        return self.background_data

    def initialize_explainer(self) -> None:
        """Initialize appropriate SHAP explainer for the model type.

        Raises:
            ValueError: If background data is required but not provided
        """
        background_array = self._prepare_background_data()

        # Try different explainer types based on model
        try:
            # TreeExplainer for tree-based models (XGBoost, LightGBM, RandomForest)
            self.explainer = shap.TreeExplainer(self.model, data=background_array)
            self.explainer_type = "tree"
            logger.info("using_tree_explainer")
        except Exception:
            try:
                # LinearExplainer for linear models
                self.explainer = shap.LinearExplainer(self.model, background_array)
                self.explainer_type = "linear"
                logger.info("using_linear_explainer")
            except Exception:
                # KernelExplainer as fallback (model-agnostic but slower)
                if background_array is None:
                    msg = "Background data required for KernelExplainer"
                    raise ValueError(msg)

                self.explainer = shap.KernelExplainer(self.model.predict, background_array)
                self.explainer_type = "kernel"
                logger.warning(
                    "using_kernel_explainer",
                    note="slower than specialized explainers",
                )

    def calculate_shap_values(
        self,
        X: pl.DataFrame | NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Calculate SHAP values for input features.

        Args:
            X: Input features (DataFrame or numpy array)

        Returns:
            Array of SHAP values (rows x features)

        Raises:
            ValueError: If explainer not initialized
        """
        if self.explainer is None:
            self.initialize_explainer()

        if self.explainer is None:
            msg = "Failed to initialize SHAP explainer"
            raise ValueError(msg)

        # Convert to numpy array if needed
        X_array = X.select(self.feature_names).to_numpy() if isinstance(X, pl.DataFrame) else X

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_array)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Multi-class case - take first class
            shap_values = shap_values[0]

        result = np.array(shap_values, dtype=np.float64)

        logger.debug(
            "calculated_shap_values",
            num_samples=result.shape[0],
            num_features=result.shape[1] if result.ndim > 1 else 1,
        )

        return result

    def get_expected_value(self) -> float:
        """Get the expected value (base value) from the explainer.

        Returns:
            Expected value (baseline prediction)

        Raises:
            ValueError: If explainer not initialized
        """
        if self.explainer is None:
            self.initialize_explainer()

        if self.explainer is None:
            msg = "Failed to initialize SHAP explainer"
            raise ValueError(msg)

        expected_value = self.explainer.expected_value

        # Handle list format (multi-class)
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(expected_value[0])
        else:
            expected_value = float(expected_value)

        return expected_value

    def aggregate_shap_importance(
        self,
        shap_values: NDArray[np.floating[Any]],
    ) -> dict[str, float]:
        """Aggregate SHAP values to feature importance scores.

        Args:
            shap_values: Array of SHAP values (rows x features)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Calculate mean absolute SHAP value per feature
        if shap_values.ndim == 1:
            # Single prediction
            mean_abs_shap = np.abs(shap_values)
        else:
            # Multiple predictions
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance = dict(
            zip(
                self.feature_names,
                mean_abs_shap.tolist(),
                strict=True,
            )
        )

        logger.debug(
            "aggregated_shap_importance",
            num_features=len(importance),
            top_feature=max(importance.items(), key=lambda x: x[1])[0],
        )

        return importance

    def calculate_single_prediction_shap(
        self,
        X: pl.DataFrame | NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], float]:
        """Calculate SHAP values for a single prediction.

        Args:
            X: Input features (single row)

        Returns:
            Tuple of (shap_values, expected_value)

        Raises:
            ValueError: If input has more than one row
        """
        # Validate single row
        if isinstance(X, pl.DataFrame):
            if len(X) != 1:
                msg = f"Expected single row, got {len(X)} rows"
                raise ValueError(msg)
        elif isinstance(X, np.ndarray) and X.shape[0] != 1:
            msg = f"Expected single row, got {X.shape[0]} rows"
            raise ValueError(msg)

        shap_values = self.calculate_shap_values(X)
        expected_value = self.get_expected_value()

        # Return first (and only) row of SHAP values
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        return shap_values, expected_value
