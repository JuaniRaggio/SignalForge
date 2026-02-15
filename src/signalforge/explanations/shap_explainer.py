"""SHAP-based explainer for ML model predictions."""

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
import shap
import structlog
from numpy.typing import NDArray

from .schemas import ExplanationConfig, FeatureContribution, PredictionExplanation

logger = structlog.get_logger(__name__)


class SHAPExplainer:
    """Generate SHAP-based explanations for ML predictions."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: pl.DataFrame | None = None,
    ) -> None:
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (sklearn, torch, etc.)
            feature_names: List of feature names in prediction order
            background_data: Background dataset for SHAP explainer (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data

        # Initialize SHAP explainer based on model type
        self._init_explainer()

        logger.info(
            "shap_explainer_initialized",
            num_features=len(feature_names),
            has_background_data=background_data is not None,
        )

    def _init_explainer(self) -> None:
        """Initialize appropriate SHAP explainer for the model type."""
        if self.background_data is not None:
            background_array = self.background_data.to_numpy()
        else:
            background_array = None

        # Try different explainer types based on model
        try:
            # TreeExplainer for tree-based models
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
                logger.warning("using_kernel_explainer", note="slower than specialized explainers")

    def explain_prediction(
        self,
        X: pl.DataFrame,
        config: ExplanationConfig | None = None,
    ) -> PredictionExplanation:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            X: Input features (single row DataFrame)
            config: Configuration for explanation generation

        Returns:
            PredictionExplanation with detailed feature contributions
        """
        if config is None:
            config = ExplanationConfig()

        if len(X) != 1:
            msg = f"Expected single row, got {len(X)} rows"
            raise ValueError(msg)

        # Select only feature columns (exclude non-numeric or metadata columns)
        X_features = X.select(self.feature_names)

        # Calculate SHAP values
        shap_values_array = self._calculate_shap_values(X_features)
        shap_values = shap_values_array[0]  # First (and only) row

        # Get feature values
        feature_values = X_features.to_numpy()[0]

        # Get prediction
        prediction = float(self.model.predict(X_features.to_numpy())[0])

        # Get base value (expected value)
        base_value = float(self.explainer.expected_value)

        # Get top features
        top_features = self._get_top_features(
            shap_values=list(shap_values),
            feature_values=list(feature_values),
            k=config.top_k_features,
            include_negative=config.include_negative,
        )

        # Calculate total contributions
        total_positive = sum(f.contribution for f in top_features if f.contribution > 0)
        total_negative = sum(f.contribution for f in top_features if f.contribution < 0)

        # Generate confidence factors
        confidence_factors = self._generate_confidence_factors(top_features, shap_values)

        # Placeholder narrative (will be filled by NarrativeGenerator)
        narrative = ""
        if config.generate_narrative:
            from .narrative import NarrativeGenerator

            generator = NarrativeGenerator()
            temp_explanation = PredictionExplanation(
                symbol=str(X.get_column("symbol")[0]) if "symbol" in X.columns else "UNKNOWN",
                prediction=prediction,
                base_value=base_value,
                top_features=top_features,
                total_positive_contribution=total_positive,
                total_negative_contribution=total_negative,
                narrative="",
                confidence_factors=confidence_factors,
                generated_at=datetime.now(),
            )
            narrative = generator.generate_narrative(temp_explanation)

        explanation = PredictionExplanation(
            symbol=str(X.get_column("symbol")[0]) if "symbol" in X.columns else "UNKNOWN",
            prediction=prediction,
            base_value=base_value,
            top_features=top_features,
            total_positive_contribution=total_positive,
            total_negative_contribution=total_negative,
            narrative=narrative,
            confidence_factors=confidence_factors,
            generated_at=datetime.now(),
        )

        logger.info(
            "generated_explanation",
            symbol=explanation.symbol,
            prediction=prediction,
            num_top_features=len(top_features),
        )

        return explanation

    def explain_batch(
        self,
        X: pl.DataFrame,
        config: ExplanationConfig | None = None,
    ) -> list[PredictionExplanation]:
        """
        Explain multiple predictions.

        Args:
            X: Input features (multiple rows)
            config: Configuration for explanation generation

        Returns:
            List of PredictionExplanation objects
        """
        explanations = []
        for i in range(len(X)):
            row = X[i : i + 1]
            explanation = self.explain_prediction(row, config)
            explanations.append(explanation)

        logger.info("generated_batch_explanations", count=len(explanations))
        return explanations

    def _calculate_shap_values(self, X: pl.DataFrame) -> NDArray[np.float64]:
        """
        Calculate SHAP values for features.

        Args:
            X: Input features (should only contain feature columns)

        Returns:
            Array of SHAP values (rows x features)
        """
        # Ensure we only use feature columns
        if set(X.columns) != set(self.feature_names):
            X = X.select(self.feature_names)

        X_array = X.to_numpy()
        shap_values = self.explainer.shap_values(X_array)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Multi-class case - take first class
            shap_values = shap_values[0]

        return np.array(shap_values)

    def _get_top_features(
        self,
        shap_values: list[float],
        feature_values: list[float],
        k: int,
        include_negative: bool = True,
    ) -> list[FeatureContribution]:
        """
        Get top K contributing features.

        Args:
            shap_values: SHAP values for each feature
            feature_values: Actual feature values
            k: Number of top features to return
            include_negative: Include negatively contributing features

        Returns:
            List of FeatureContribution objects, ranked by importance
        """
        # Create feature contributions
        contributions: list[dict[str, Any]] = []
        for name, value, shap_val in zip(
            self.feature_names, feature_values, shap_values, strict=True
        ):
            direction = "positive" if shap_val > 0 else "negative"

            # Skip negative if not included
            if not include_negative and shap_val < 0:
                continue

            human_readable = self._format_feature_description(name, value, shap_val)

            contributions.append(
                {
                    "feature_name": name,
                    "feature_value": float(value),
                    "contribution": float(shap_val),
                    "direction": direction,
                    "abs_contribution": abs(float(shap_val)),
                    "human_readable": human_readable,
                }
            )

        # Sort by absolute contribution
        contributions.sort(key=lambda x: float(x["abs_contribution"]), reverse=True)

        # Take top K
        top_k = contributions[:k]

        # Add importance rank
        result = []
        for rank, contrib in enumerate(top_k, start=1):
            result.append(
                FeatureContribution(
                    feature_name=contrib["feature_name"],
                    feature_value=contrib["feature_value"],
                    contribution=contrib["contribution"],
                    direction=contrib["direction"],
                    importance_rank=rank,
                    human_readable=contrib["human_readable"],
                )
            )

        return result

    def _format_feature_description(
        self,
        feature_name: str,
        feature_value: float,
        contribution: float,
    ) -> str:
        """
        Format human-readable feature description.

        Args:
            feature_name: Name of the feature
            feature_value: Actual value
            contribution: SHAP contribution

        Returns:
            Human-readable description
        """
        # Calculate percentage impact
        if abs(self.explainer.expected_value) > 0:
            pct_impact = (contribution / abs(self.explainer.expected_value)) * 100
        else:
            pct_impact = 0

        direction = "up" if contribution > 0 else "down"

        # Format based on feature type
        if "rsi" in feature_name.lower():
            if feature_value > 70:
                condition = "(overbought)"
            elif feature_value < 30:
                condition = "(oversold)"
            else:
                condition = "(neutral)"
            return f"{feature_name} = {feature_value:.1f} {condition} -> {direction} by {abs(pct_impact):.1f}%"

        elif "macd" in feature_name.lower():
            condition = "(bullish)" if feature_value > 0 else "(bearish)"
            return f"{feature_name} = {feature_value:.3f} {condition} -> {direction} by {abs(pct_impact):.1f}%"

        elif "volume" in feature_name.lower():
            condition = "(high)" if feature_value > 1.5 else "(normal)" if feature_value > 0.5 else "(low)"
            return f"{feature_name} = {feature_value:.2f} {condition} -> {direction} by {abs(pct_impact):.1f}%"

        else:
            # Generic format
            return f"{feature_name} = {feature_value:.4f} -> {direction} by {abs(pct_impact):.1f}%"

    def _generate_confidence_factors(
        self,
        top_features: list[FeatureContribution],
        all_shap_values: list[float] | NDArray[np.float64],
    ) -> list[str]:
        """
        Generate factors affecting prediction confidence.

        Args:
            top_features: Top contributing features
            all_shap_values: All SHAP values

        Returns:
            List of confidence factor descriptions
        """
        factors = []

        # Check alignment of top features
        if len(top_features) >= 3:
            top_directions = [f.direction for f in top_features[:3]]
            if len(set(top_directions)) == 1:
                factors.append("High agreement among top features")
            else:
                factors.append("Mixed signals from top features")

        # Check magnitude of contributions
        if top_features:
            max_contrib = max(abs(f.contribution) for f in top_features)
            avg_contrib = sum(abs(f.contribution) for f in top_features) / len(top_features)

            if max_contrib > 3 * avg_contrib:
                factors.append("Single dominant feature detected")
            else:
                factors.append("Balanced feature contributions")

        # Check overall feature variance
        shap_array = np.array(all_shap_values)
        shap_std = float(np.std(shap_array))
        if shap_std > 0.5:
            factors.append("High feature importance variance")
        else:
            factors.append("Low feature importance variance")

        return factors

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get global feature importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.background_data is None:
            msg = "Background data required for global feature importance"
            raise ValueError(msg)

        # Calculate SHAP values for background data
        shap_values = self._calculate_shap_values(self.background_data)

        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance = dict(zip(self.feature_names, mean_abs_shap.tolist(), strict=True))

        logger.info("calculated_global_feature_importance", num_features=len(importance))
        return importance

    def compare_predictions(
        self,
        X1: pl.DataFrame,
        X2: pl.DataFrame,
    ) -> dict[str, Any]:
        """
        Compare explanations between two predictions.

        Args:
            X1: First prediction features
            X2: Second prediction features

        Returns:
            Dictionary with comparison details
        """
        exp1 = self.explain_prediction(X1, ExplanationConfig(generate_narrative=False))
        exp2 = self.explain_prediction(X2, ExplanationConfig(generate_narrative=False))

        # Calculate prediction difference
        pred_diff = exp2.prediction - exp1.prediction

        # Find features with biggest SHAP changes
        feature_changes: list[dict[str, Any]] = []
        shap1_dict = {f.feature_name: f.contribution for f in exp1.top_features}
        shap2_dict = {f.feature_name: f.contribution for f in exp2.top_features}

        all_features = set(shap1_dict.keys()) | set(shap2_dict.keys())
        for feature in all_features:
            shap1_val = shap1_dict.get(feature, 0.0)
            shap2_val = shap2_dict.get(feature, 0.0)
            change = shap2_val - shap1_val

            feature_changes.append(
                {
                    "feature": feature,
                    "shap_change": change,
                    "shap_1": shap1_val,
                    "shap_2": shap2_val,
                }
            )

        # Sort by absolute change
        feature_changes.sort(key=lambda x: abs(float(x["shap_change"])), reverse=True)

        comparison = {
            "prediction_1": exp1.prediction,
            "prediction_2": exp2.prediction,
            "prediction_difference": pred_diff,
            "base_value_1": exp1.base_value,
            "base_value_2": exp2.base_value,
            "top_feature_changes": feature_changes[:5],
            "symbol_1": exp1.symbol,
            "symbol_2": exp2.symbol,
        }

        logger.info(
            "compared_predictions",
            pred_diff=pred_diff,
            num_feature_changes=len(feature_changes),
        )

        return comparison
