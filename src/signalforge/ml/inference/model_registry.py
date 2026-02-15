"""Model registry for managing trained models.

This module provides a centralized registry for storing, retrieving, and
managing trained ML models. It supports:
- Model versioning and metadata tracking
- Performance metrics storage
- Default model selection
- Tag-based model categorization
- Best model selection by metric

The registry stores models on disk and maintains an index of metadata.

Key Classes:
    ModelInfo: Metadata for a registered model
    ModelRegistry: Central registry for model management

Examples:
    Basic usage:

    >>> from signalforge.ml.inference import ModelRegistry
    >>> from signalforge.ml.models import LSTMPredictor
    >>>
    >>> # Create registry
    >>> registry = ModelRegistry("models/")
    >>>
    >>> # Register a model
    >>> model = LSTMPredictor()
    >>> model_id = registry.register(
    ...     model,
    ...     metrics={"mse": 0.025, "mae": 0.12},
    ...     tags={"type": "lstm", "dataset": "sp500"}
    ... )
    >>>
    >>> # Set as default
    >>> registry.set_default(model_id)
    >>>
    >>> # Retrieve model
    >>> loaded_model = registry.get(model_id)
    >>> default_model = registry.get_default()
    >>>
    >>> # Find best model
    >>> best_model = registry.get_best(metric="mse")
"""

from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.ml.models.base import BasePredictor

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered model.

    Attributes:
        model_id: Unique identifier for the model
        model_name: Human-readable name (from model.model_name)
        model_version: Version string (from model.model_version)
        created_at: Timestamp when model was registered
        metrics: Performance metrics (e.g., {"mse": 0.025, "r2": 0.85})
        tags: Arbitrary tags for categorization
        is_default: Whether this is the default model
        file_path: Path to serialized model file
        status: Model status (staging, production, archived)
        traffic_percentage: Traffic percentage for A/B testing (0.0-1.0)
        onnx_path: Optional path to ONNX version of model

    Examples:
        >>> info = ModelInfo(
        ...     model_id="abc-123",
        ...     model_name="LSTMPredictor",
        ...     model_version="1.0.0",
        ...     created_at=datetime.now(),
        ...     metrics={"mse": 0.025},
        ...     tags={"type": "lstm"},
        ...     is_default=True,
        ...     file_path="models/abc-123.pkl",
        ...     status="production",
        ...     traffic_percentage=1.0,
        ...     onnx_path="models/abc-123.onnx"
        ... )
    """

    model_id: str
    model_name: str
    model_version: str
    created_at: datetime
    metrics: dict[str, float]
    tags: dict[str, str]
    is_default: bool
    file_path: str
    status: str = "staging"
    traffic_percentage: float = 0.0
    onnx_path: str | None = None


class ModelRegistry:
    """Registry for managing trained models.

    This class provides a centralized location for storing and retrieving
    trained models. It maintains metadata about each model including
    performance metrics, tags, and version information.

    The registry stores:
    - Serialized model files (using pickle or model-specific format)
    - Metadata index (JSON file with all model information)
    - Default model reference

    Attributes:
        storage_path: Root directory for model storage
        index_file: Path to metadata index file
        models_index: In-memory index of model metadata
    """

    def __init__(self, storage_path: str = "models/") -> None:
        """Initialize model registry.

        Creates storage directory if it doesn't exist and loads
        the existing model index.

        Args:
            storage_path: Root directory for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.storage_path / "registry_index.json"
        self.models_index: dict[str, ModelInfo] = {}

        # Load existing index
        self._load_index()

        logger.info(
            "model_registry_initialized",
            storage_path=str(self.storage_path),
            num_models=len(self.models_index),
        )

    def register(
        self,
        model: BasePredictor,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        status: str = "staging",
        onnx_path: str | None = None,
    ) -> str:
        """Register a trained model.

        Saves the model to disk and records its metadata in the registry.

        Args:
            model: Trained model to register
            metrics: Performance metrics (e.g., {"mse": 0.025, "r2": 0.85})
            tags: Arbitrary tags for categorization
            status: Model status (staging, production, archived)
            onnx_path: Optional path to ONNX version

        Returns:
            Unique model ID

        Raises:
            ValueError: If model is invalid or status is invalid
            RuntimeError: If model saving fails

        Examples:
            >>> registry = ModelRegistry()
            >>> model = LSTMPredictor()
            >>> model_id = registry.register(
            ...     model,
            ...     metrics={"mse": 0.025, "mae": 0.12},
            ...     tags={"type": "lstm", "dataset": "sp500"},
            ...     status="staging"
            ... )
            >>> print(f"Model registered with ID: {model_id}")
        """
        # Validate status
        valid_statuses = {"staging", "production", "archived"}
        if status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got {status}"
            )

        # Generate unique ID
        model_id = str(uuid.uuid4())

        # Determine file path
        file_path = self.storage_path / f"{model_id}.pkl"

        # Save model
        try:
            with open(file_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(
                "model_saved_to_disk",
                model_id=model_id,
                file_path=str(file_path),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}") from e

        # Create model info
        info = ModelInfo(
            model_id=model_id,
            model_name=model.model_name,
            model_version=model.model_version,
            created_at=datetime.now(),
            metrics=metrics or {},
            tags=tags or {},
            is_default=False,
            file_path=str(file_path),
            status=status,
            traffic_percentage=0.0,
            onnx_path=onnx_path,
        )

        # Add to index
        self.models_index[model_id] = info

        # Save index
        self._save_index()

        logger.info(
            "model_registered",
            model_id=model_id,
            model_name=info.model_name,
            model_version=info.model_version,
            metrics=metrics,
            tags=tags,
            status=status,
        )

        return model_id

    def get(self, model_id: str) -> BasePredictor:
        """Load a model by ID.

        Args:
            model_id: Unique model identifier

        Returns:
            Loaded model instance

        Raises:
            KeyError: If model_id not found
            RuntimeError: If model loading fails

        Examples:
            >>> model = registry.get("abc-123-def-456")
            >>> predictions = model.predict(X)
        """
        if model_id not in self.models_index:
            raise KeyError(f"Model not found: {model_id}")

        info = self.models_index[model_id]
        file_path = Path(info.file_path)

        if not file_path.exists():
            raise RuntimeError(f"Model file not found: {file_path}")

        try:
            with open(file_path, "rb") as f:
                model: BasePredictor = pickle.load(f)
            logger.debug(
                "model_loaded_from_disk",
                model_id=model_id,
                model_name=info.model_name,
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def get_best(
        self,
        metric: str = "mse",
        model_type: str | None = None,
    ) -> BasePredictor:
        """Get best performing model by metric.

        Args:
            metric: Metric name to optimize (lower is better)
            model_type: Optional filter by model type tag

        Returns:
            Best performing model

        Raises:
            ValueError: If no models match criteria
            KeyError: If metric not found in any model

        Examples:
            >>> # Get overall best model by MSE
            >>> best_model = registry.get_best(metric="mse")
            >>>
            >>> # Get best LSTM model
            >>> best_lstm = registry.get_best(
            ...     metric="mse",
            ...     model_type="lstm"
            ... )
        """
        # Filter models
        candidates_list = list(self.models_index.values())

        if model_type is not None:
            candidates_list = [
                m for m in candidates_list
                if m.tags.get("type") == model_type
            ]

        if not candidates_list:
            raise ValueError(
                f"No models found"
                f"{' for type: ' + model_type if model_type else ''}"
            )

        # Filter models that have the metric
        candidates_with_metric = [
            m for m in candidates_list
            if metric in m.metrics
        ]

        if not candidates_with_metric:
            raise KeyError(f"No models found with metric: {metric}")

        # Find best (lowest metric value)
        best_info = min(
            candidates_with_metric,
            key=lambda m: m.metrics[metric],
        )

        logger.info(
            "best_model_selected",
            model_id=best_info.model_id,
            model_name=best_info.model_name,
            metric=metric,
            value=best_info.metrics[metric],
        )

        return self.get(best_info.model_id)

    def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """List all registered models.

        Args:
            model_type: Optional filter by model type tag

        Returns:
            List of ModelInfo objects, sorted by creation time (newest first)

        Examples:
            >>> # List all models
            >>> all_models = registry.list_models()
            >>> for info in all_models:
            ...     print(f"{info.model_name} v{info.model_version}")
            >>>
            >>> # List only LSTM models
            >>> lstm_models = registry.list_models(model_type="lstm")
        """
        models_list = list(self.models_index.values())

        if model_type is not None:
            models_list = [m for m in models_list if m.tags.get("type") == model_type]

        # Sort by creation time, newest first
        models_list.sort(key=lambda m: m.created_at, reverse=True)

        return models_list

    def set_default(self, model_id: str) -> None:
        """Set default model for predictions.

        Args:
            model_id: ID of model to set as default

        Raises:
            KeyError: If model_id not found

        Examples:
            >>> registry.set_default("abc-123-def-456")
        """
        if model_id not in self.models_index:
            raise KeyError(f"Model not found: {model_id}")

        # Unset current default
        for info in self.models_index.values():
            info.is_default = False

        # Set new default
        self.models_index[model_id].is_default = True

        # Save index
        self._save_index()

        logger.info(
            "default_model_updated",
            model_id=model_id,
            model_name=self.models_index[model_id].model_name,
        )

    def get_default(self) -> BasePredictor:
        """Get default model.

        Returns:
            Default model instance

        Raises:
            ValueError: If no default model is set

        Examples:
            >>> model = registry.get_default()
            >>> predictions = model.predict(X)
        """
        for model_id, info in self.models_index.items():
            if info.is_default:
                return self.get(model_id)

        raise ValueError("No default model set")

    def delete(self, model_id: str) -> None:
        """Delete a model from the registry.

        Args:
            model_id: ID of model to delete

        Raises:
            KeyError: If model_id not found
            ValueError: If trying to delete the default model

        Examples:
            >>> registry.delete("abc-123-def-456")
        """
        if model_id not in self.models_index:
            raise KeyError(f"Model not found: {model_id}")

        info = self.models_index[model_id]

        if info.is_default:
            raise ValueError(
                "Cannot delete default model. Set a different default first."
            )

        # Delete file
        file_path = Path(info.file_path)
        if file_path.exists():
            file_path.unlink()

        # Remove from index
        del self.models_index[model_id]

        # Save index
        self._save_index()

        logger.info(
            "model_deleted",
            model_id=model_id,
            model_name=info.model_name,
        )

    def _load_index(self) -> None:
        """Load model index from disk."""
        if not self.index_file.exists():
            logger.info("no_existing_registry_index_found")
            return

        try:
            with open(self.index_file) as f:
                data = json.load(f)

            # Convert to ModelInfo objects
            for model_id, model_data in data.items():
                # Parse datetime
                model_data["created_at"] = datetime.fromisoformat(
                    model_data["created_at"]
                )
                self.models_index[model_id] = ModelInfo(**model_data)

            logger.info(
                "registry_index_loaded",
                num_models=len(self.models_index),
            )
        except Exception as e:
            logger.error(
                "failed_to_load_registry_index",
                error=str(e),
            )
            # Continue with empty index

    def _save_index(self) -> None:
        """Save model index to disk."""
        try:
            # Convert to serializable format
            data = {}
            for model_id, info in self.models_index.items():
                info_dict = asdict(info)
                # Convert datetime to ISO format
                info_dict["created_at"] = info.created_at.isoformat()
                data[model_id] = info_dict

            with open(self.index_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(
                "registry_index_saved",
                num_models=len(self.models_index),
            )
        except Exception as e:
            logger.error(
                "failed_to_save_registry_index",
                error=str(e),
            )

    def promote_to_production(
        self,
        model_id: str,
        traffic_percentage: float = 1.0,
    ) -> None:
        """Promote a model to production status.

        Args:
            model_id: ID of model to promote
            traffic_percentage: Initial traffic percentage (0.0-1.0)

        Raises:
            KeyError: If model_id not found
            ValueError: If traffic_percentage is invalid

        Examples:
            >>> # Promote to production with 100% traffic
            >>> registry.promote_to_production("abc-123", 1.0)
            >>>
            >>> # Canary deployment with 5% traffic
            >>> registry.promote_to_production("def-456", 0.05)
        """
        if model_id not in self.models_index:
            raise KeyError(f"Model not found: {model_id}")

        if not (0.0 <= traffic_percentage <= 1.0):
            raise ValueError(
                f"traffic_percentage must be 0.0-1.0, got {traffic_percentage}"
            )

        self.models_index[model_id].status = "production"
        self.models_index[model_id].traffic_percentage = traffic_percentage

        self._save_index()

        logger.info(
            "model_promoted_to_production",
            model_id=model_id,
            model_name=self.models_index[model_id].model_name,
            traffic_percentage=traffic_percentage,
        )

    def set_traffic_percentage(
        self,
        model_id: str,
        traffic_percentage: float,
    ) -> None:
        """Set traffic percentage for a production model.

        Args:
            model_id: Model identifier
            traffic_percentage: Traffic percentage (0.0-1.0)

        Raises:
            KeyError: If model_id not found
            ValueError: If traffic_percentage is invalid

        Examples:
            >>> # Gradually increase traffic during canary deployment
            >>> registry.set_traffic_percentage("abc-123", 0.10)  # 10%
            >>> # Monitor metrics...
            >>> registry.set_traffic_percentage("abc-123", 0.50)  # 50%
        """
        if model_id not in self.models_index:
            raise KeyError(f"Model not found: {model_id}")

        if not (0.0 <= traffic_percentage <= 1.0):
            raise ValueError(
                f"traffic_percentage must be 0.0-1.0, got {traffic_percentage}"
            )

        self.models_index[model_id].traffic_percentage = traffic_percentage
        self._save_index()

        logger.info(
            "model_traffic_updated",
            model_id=model_id,
            traffic_percentage=traffic_percentage,
        )

    def archive_model(self, model_id: str) -> None:
        """Archive a model (set status to archived, traffic to 0).

        Args:
            model_id: Model identifier

        Raises:
            KeyError: If model_id not found

        Examples:
            >>> registry.archive_model("old-model-123")
        """
        if model_id not in self.models_index:
            raise KeyError(f"Model not found: {model_id}")

        self.models_index[model_id].status = "archived"
        self.models_index[model_id].traffic_percentage = 0.0
        self.models_index[model_id].is_default = False

        self._save_index()

        logger.info(
            "model_archived",
            model_id=model_id,
            model_name=self.models_index[model_id].model_name,
        )

    def get_production_models(self) -> list[ModelInfo]:
        """Get all models in production status.

        Returns:
            List of ModelInfo for production models

        Examples:
            >>> prod_models = registry.get_production_models()
            >>> for model in prod_models:
            ...     print(f"{model.model_name}: {model.traffic_percentage*100}% traffic")
        """
        return [
            info
            for info in self.models_index.values()
            if info.status == "production"
        ]

    def list_versions(self, model_name: str) -> list[ModelInfo]:
        """List all versions of a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of ModelInfo for all versions, sorted newest first

        Examples:
            >>> versions = registry.list_versions("LSTMPredictor")
            >>> for v in versions:
            ...     print(f"v{v.model_version}: {v.status} - {v.metrics.get('mse', 0):.4f} MSE")
        """
        versions = [
            info
            for info in self.models_index.values()
            if info.model_name == model_name
        ]

        # Sort by creation time, newest first
        versions.sort(key=lambda m: m.created_at, reverse=True)

        return versions


__all__ = [
    "ModelInfo",
    "ModelRegistry",
]
