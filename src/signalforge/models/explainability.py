"""SQLAlchemy models for ML model explainability and feature importance tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Uuid, func
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base


class ImportanceType(str, Enum):
    """Type of feature importance calculation."""

    GAIN = "gain"
    PERMUTATION = "permutation"
    SHAP_MEAN = "shap_mean"


class PredictionExplanation(Base):
    """Store SHAP explanations for individual predictions.

    Tracks detailed feature contributions for each prediction to enable
    interpretability and debugging of model decisions.
    """

    __tablename__ = "prediction_explanations"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique explanation ID",
    )
    prediction_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        index=True,
        comment="Foreign key to predictions table",
    )
    model_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        comment="Identifier for the model used",
    )
    symbol: Mapped[str] = mapped_column(
        String(20),
        index=True,
        comment="Stock symbol",
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        comment="Prediction timestamp",
    )
    shap_values: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Dictionary of feature_name: shap_value",
    )
    base_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 6),
        nullable=False,
        comment="SHAP base value (expected value)",
    )
    prediction_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 6),
        nullable=False,
        comment="Predicted value",
    )
    top_features: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of top contributing features with values",
    )
    explanation_text: Mapped[str] = mapped_column(
        String(2000),
        nullable=False,
        comment="Human-readable explanation of prediction",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_prediction_explanations_model_timestamp", "model_id", "timestamp"),
        Index("ix_prediction_explanations_symbol_timestamp", "symbol", "timestamp"),
    )


class FeatureImportanceHistory(Base):
    """Track feature importance over time.

    Records how feature importance changes across different time periods,
    enabling trend analysis and feature stability monitoring.
    """

    __tablename__ = "feature_importance_history"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique history record ID",
    )
    model_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        comment="Identifier for the model",
    )
    feature_name: Mapped[str] = mapped_column(
        String(200),
        index=True,
        comment="Name of the feature",
    )
    importance_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 6),
        nullable=False,
        comment="Calculated importance value",
    )
    importance_type: Mapped[ImportanceType] = mapped_column(
        SQLEnum(ImportanceType, native_enum=False, length=50),
        index=True,
        nullable=False,
        comment="Type of importance calculation",
    )
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
        comment="When importance was calculated",
    )
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Start of analysis period",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of analysis period",
    )

    __table_args__ = (
        Index(
            "ix_feature_importance_model_feature_type",
            "model_id",
            "feature_name",
            "importance_type",
        ),
        Index("ix_feature_importance_model_calculated", "model_id", "calculated_at"),
        Index("ix_feature_importance_period", "period_start", "period_end"),
    )


class ModelAccuracyByFeature(Base):
    """Track which features correlate with accurate predictions.

    Analyzes prediction accuracy when specific features have high vs low
    importance, revealing feature reliability and predictive power.
    """

    __tablename__ = "model_accuracy_by_feature"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique accuracy record ID",
    )
    model_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        comment="Identifier for the model",
    )
    feature_name: Mapped[str] = mapped_column(
        String(200),
        index=True,
        comment="Name of the feature",
    )
    accuracy_when_high: Mapped[Decimal] = mapped_column(
        Numeric(8, 6),
        nullable=False,
        comment="Accuracy when feature importance is high (0-1 scale)",
    )
    accuracy_when_low: Mapped[Decimal] = mapped_column(
        Numeric(8, 6),
        nullable=False,
        comment="Accuracy when feature importance is low (0-1 scale)",
    )
    sample_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of predictions analyzed",
    )
    correlation_coefficient: Mapped[Decimal] = mapped_column(
        Numeric(8, 6),
        nullable=False,
        comment="Correlation between feature importance and accuracy",
    )
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
        comment="When analysis was performed",
    )

    __table_args__ = (
        Index("ix_model_accuracy_model_feature", "model_id", "feature_name"),
        Index("ix_model_accuracy_calculated", "calculated_at"),
    )


class EnsembleModelComparison(Base):
    """Compare ensemble model contributions.

    Tracks how individual models within an ensemble contribute to final
    predictions, including weights, confidence, and impact.
    """

    __tablename__ = "ensemble_model_comparisons"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique comparison record ID",
    )
    ensemble_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        comment="Identifier for the ensemble",
    )
    prediction_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        index=True,
        comment="Foreign key to predictions table",
    )
    model_name: Mapped[str] = mapped_column(
        String(100),
        index=True,
        comment="Name of the individual model in ensemble",
    )
    model_weight: Mapped[Decimal] = mapped_column(
        Numeric(8, 6),
        nullable=False,
        comment="Weight of this model in ensemble (0-1 scale)",
    )
    model_prediction: Mapped[Decimal] = mapped_column(
        Numeric(14, 6),
        nullable=False,
        comment="Individual model prediction value",
    )
    model_confidence: Mapped[Decimal] = mapped_column(
        Numeric(8, 6),
        nullable=False,
        comment="Confidence score from this model (0-1 scale)",
    )
    contribution_to_final: Mapped[Decimal] = mapped_column(
        Numeric(14, 6),
        nullable=False,
        comment="Weighted contribution to final ensemble prediction",
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
        comment="Prediction timestamp",
    )

    __table_args__ = (
        Index(
            "ix_ensemble_comparison_ensemble_prediction",
            "ensemble_id",
            "prediction_id",
        ),
        Index("ix_ensemble_comparison_ensemble_timestamp", "ensemble_id", "timestamp"),
        Index("ix_ensemble_comparison_model_timestamp", "model_name", "timestamp"),
    )
