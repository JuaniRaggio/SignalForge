"""Visualization data generation for explanations."""

from pathlib import Path
from typing import Any

import structlog

from .schemas import PredictionExplanation

logger = structlog.get_logger(__name__)


class ExplanationVisualizer:
    """Generate visualization data for explanations."""

    def __init__(self) -> None:
        """Initialize explanation visualizer."""
        logger.info("explanation_visualizer_initialized")

    def generate_waterfall_data(
        self,
        explanation: PredictionExplanation,
    ) -> dict[str, Any]:
        """
        Generate data for waterfall chart.

        Shows how features incrementally contribute from base value to final prediction.

        Args:
            explanation: The prediction explanation

        Returns:
            JSON-serializable dict for frontend waterfall chart
        """
        waterfall_data: dict[str, Any] = {
            "base_value": explanation.base_value,
            "final_prediction": explanation.prediction,
            "features": [],
        }

        # Start with base value
        cumulative_value = explanation.base_value

        # Add each feature's contribution
        for feature in explanation.top_features:
            waterfall_data["features"].append(
                {
                    "name": feature.feature_name,
                    "value": feature.feature_value,
                    "contribution": feature.contribution,
                    "start": cumulative_value,
                    "end": cumulative_value + feature.contribution,
                    "direction": feature.direction,
                    "label": feature.human_readable,
                }
            )
            cumulative_value += feature.contribution

        # Add residual if doesn't sum to prediction
        residual = explanation.prediction - cumulative_value
        if abs(residual) > 1e-6:
            waterfall_data["features"].append(
                {
                    "name": "Other Features",
                    "value": 0.0,
                    "contribution": residual,
                    "start": cumulative_value,
                    "end": explanation.prediction,
                    "direction": "positive" if residual > 0 else "negative",
                    "label": f"Other features -> {residual:+.4f}",
                }
            )

        logger.info(
            "generated_waterfall_data",
            symbol=explanation.symbol,
            num_features=len(waterfall_data["features"]),
        )

        return waterfall_data

    def generate_force_plot_data(
        self,
        explanation: PredictionExplanation,
    ) -> dict[str, Any]:
        """
        Generate data for SHAP force plot.

        Shows positive and negative forces pushing prediction away from base value.

        Args:
            explanation: The prediction explanation

        Returns:
            JSON-serializable dict for frontend force plot
        """
        positive_features = [f for f in explanation.top_features if f.contribution > 0]
        negative_features = [f for f in explanation.top_features if f.contribution < 0]

        force_plot_data: dict[str, Any] = {
            "base_value": explanation.base_value,
            "prediction": explanation.prediction,
            "positive_forces": [
                {
                    "name": f.feature_name,
                    "value": f.feature_value,
                    "contribution": f.contribution,
                    "label": f.human_readable,
                }
                for f in positive_features
            ],
            "negative_forces": [
                {
                    "name": f.feature_name,
                    "value": f.feature_value,
                    "contribution": abs(f.contribution),
                    "label": f.human_readable,
                }
                for f in negative_features
            ],
            "total_positive": explanation.total_positive_contribution,
            "total_negative": abs(explanation.total_negative_contribution),
        }

        logger.info(
            "generated_force_plot_data",
            symbol=explanation.symbol,
            positive_count=len(positive_features),
            negative_count=len(negative_features),
        )

        return force_plot_data

    def generate_summary_plot_data(
        self,
        explanations: list[PredictionExplanation],
    ) -> dict[str, Any]:
        """
        Generate data for summary plot (multiple predictions).

        Shows feature importance across multiple predictions.

        Args:
            explanations: List of prediction explanations

        Returns:
            JSON-serializable dict for frontend summary plot
        """
        if not explanations:
            return {
                "features": [],
                "predictions": [],
                "num_predictions": 0,
            }

        # Aggregate feature contributions across all predictions
        feature_contributions: dict[str, list[float]] = {}
        symbols = []

        for exp in explanations:
            symbols.append(exp.symbol)
            for feature in exp.top_features:
                if feature.feature_name not in feature_contributions:
                    feature_contributions[feature.feature_name] = []
                feature_contributions[feature.feature_name].append(feature.contribution)

        # Calculate statistics for each feature
        feature_stats: list[dict[str, Any]] = []
        for feature_name, contributions in feature_contributions.items():
            mean_contrib = sum(contributions) / len(contributions)
            abs_mean = abs(mean_contrib)

            feature_stats.append(
                {
                    "name": feature_name,
                    "mean_contribution": mean_contrib,
                    "abs_mean_contribution": abs_mean,
                    "min_contribution": min(contributions),
                    "max_contribution": max(contributions),
                    "appearance_count": len(contributions),
                }
            )

        # Sort by absolute mean contribution
        feature_stats.sort(key=lambda x: float(x["abs_mean_contribution"]), reverse=True)

        summary_data: dict[str, Any] = {
            "features": feature_stats,
            "symbols": symbols,
            "num_predictions": len(explanations),
            "avg_prediction": sum(exp.prediction for exp in explanations) / len(explanations),
            "avg_base_value": sum(exp.base_value for exp in explanations) / len(explanations),
        }

        logger.info(
            "generated_summary_plot_data",
            num_predictions=len(explanations),
            num_features=len(feature_stats),
        )

        return summary_data

    def generate_feature_importance_chart(
        self,
        feature_importance: dict[str, float],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Generate feature importance bar chart data.

        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            top_k: Number of top features to include

        Returns:
            JSON-serializable dict for frontend bar chart
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Take top K
        top_features = sorted_features[:top_k]

        chart_data: dict[str, Any] = {
            "features": [
                {
                    "name": name,
                    "importance": importance,
                    "rank": i + 1,
                }
                for i, (name, importance) in enumerate(top_features)
            ],
            "total_features": len(feature_importance),
            "top_k": top_k,
        }

        logger.info(
            "generated_feature_importance_chart",
            total_features=len(feature_importance),
            top_k=top_k,
        )

        return chart_data

    def export_to_html(
        self,
        explanation: PredictionExplanation,
        output_path: str,
    ) -> str:
        """
        Export explanation to standalone HTML.

        Args:
            explanation: The prediction explanation
            output_path: Path to save HTML file

        Returns:
            Path to generated HTML file
        """
        # Generate visualization data
        waterfall_data = self.generate_waterfall_data(explanation)
        force_plot_data = self.generate_force_plot_data(explanation)

        # Create HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Explanation - {symbol}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .feature {{
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #3498db;
            background-color: #ecf0f1;
        }}
        .positive {{ border-left-color: #27ae60; }}
        .negative {{ border-left-color: #e74c3c; }}
        .metric {{
            display: inline-block;
            margin-right: 20px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Prediction Explanation: {symbol}</h1>
        <p>Generated: {generated_at}</p>
    </div>

    <div class="section">
        <h2>Prediction Summary</h2>
        <div class="metric">
            <span class="metric-label">Prediction:</span>
            <span class="metric-value">{prediction:.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Base Value:</span>
            <span class="metric-value">{base_value:.4f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Net Contribution:</span>
            <span class="metric-value">{net_contribution:.4f}</span>
        </div>
    </div>

    <div class="section">
        <h2>Narrative</h2>
        <p>{narrative}</p>
    </div>

    <div class="section">
        <h2>Top Contributing Features</h2>
        {features_html}
    </div>

    <div class="section">
        <h2>Confidence Factors</h2>
        <ul>
        {confidence_factors_html}
        </ul>
    </div>

    <div class="section">
        <h2>Visualization Data</h2>
        <h3>Waterfall Chart Data</h3>
        <pre>{waterfall_json}</pre>
        <h3>Force Plot Data</h3>
        <pre>{force_plot_json}</pre>
    </div>
</body>
</html>
"""

        # Generate features HTML
        features_html = ""
        for feature in explanation.top_features:
            direction_class = feature.direction
            features_html += f"""
        <div class="feature {direction_class}">
            <strong>{feature.feature_name}</strong> (Rank #{feature.importance_rank})<br>
            {feature.human_readable}<br>
            Value: {feature.feature_value:.4f}, Contribution: {feature.contribution:+.4f}
        </div>
"""

        # Generate confidence factors HTML
        confidence_factors_html = "\n".join(
            f"            <li>{factor}</li>" for factor in explanation.confidence_factors
        )

        # Fill template
        import json

        html_content = html_template.format(
            symbol=explanation.symbol,
            generated_at=explanation.generated_at.isoformat(),
            prediction=explanation.prediction,
            base_value=explanation.base_value,
            net_contribution=explanation.prediction - explanation.base_value,
            narrative=explanation.narrative,
            features_html=features_html,
            confidence_factors_html=confidence_factors_html,
            waterfall_json=json.dumps(waterfall_data, indent=2),
            force_plot_json=json.dumps(force_plot_data, indent=2),
        )

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content)

        logger.info(
            "exported_to_html",
            symbol=explanation.symbol,
            output_path=output_path,
        )

        return str(output_file.absolute())
