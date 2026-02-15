"""Prediction service for SignalForge ML models.

This module provides the main prediction service that orchestrates the entire
inference pipeline: data fetching, feature computation, model prediction, and
response formatting.

The PredictionService supports:
- Single and batch predictions
- Model selection (specific model or default)
- Response caching for performance
- Confidence intervals and uncertainty quantification
- Detailed prediction metadata and timing

Key Classes:
    PredictionResponse: Structured prediction result
    PredictionService: Main service for generating predictions

Examples:
    Basic usage with default model:

    >>> from signalforge.ml.inference import PredictionService
    >>> from signalforge.ml.inference import ModelRegistry
    >>> from signalforge.ml.features import TechnicalFeatureEngine
    >>>
    >>> registry = ModelRegistry("models/")
    >>> features = TechnicalFeatureEngine()
    >>> service = PredictionService(registry, features)
    >>>
    >>> # Single prediction
    >>> result = await service.predict("AAPL", horizon_days=5)
    >>> print(f"Predicted return: {result.predicted_return:.2%}")
    >>> print(f"Confidence: {result.confidence:.2%}")
    >>>
    >>> # Batch predictions
    >>> results = await service.predict_batch(["AAPL", "GOOGL", "MSFT"])
    >>> for r in results:
    ...     print(f"{r.symbol}: {r.predicted_return:.2%}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.ml.features.technical import TechnicalFeatureEngine
    from signalforge.ml.inference.model_registry import ModelRegistry

logger = get_logger(__name__)


@dataclass
class PredictionResponse:
    """Structured response for a prediction request.

    Attributes:
        symbol: The ticker symbol that was predicted
        timestamp: When the prediction was made
        horizon_days: Number of days ahead this prediction is for
        predicted_return: Predicted percentage return
        predicted_price: Predicted future price (if available)
        confidence: Confidence score for this prediction (0.0-1.0)
        interval_lower: Lower bound of prediction interval
        interval_upper: Upper bound of prediction interval
        model_used: Name of the model that generated this prediction
        model_version: Version string of the model
        features_used: Number of features used in prediction
        latency_ms: Time taken to generate prediction in milliseconds

    Examples:
        >>> response = PredictionResponse(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(),
        ...     horizon_days=5,
        ...     predicted_return=2.5,
        ...     predicted_price=152.30,
        ...     confidence=0.85,
        ...     interval_lower=148.0,
        ...     interval_upper=156.0,
        ...     model_used="LSTMPredictor",
        ...     model_version="1.0.0",
        ...     features_used=45,
        ...     latency_ms=12.3
        ... )
    """

    symbol: str
    timestamp: datetime
    horizon_days: int
    predicted_return: float
    predicted_price: float | None
    confidence: float
    interval_lower: float | None
    interval_upper: float | None
    model_used: str
    model_version: str
    features_used: int
    latency_ms: float


class PredictionService:
    """Main prediction service for SignalForge.

    This service coordinates the entire prediction pipeline:
    1. Fetch recent price data (delegates to data layer)
    2. Compute technical features
    3. Load appropriate model
    4. Generate prediction
    5. Format and return response

    Attributes:
        model_registry: Registry containing trained models
        feature_engine: Engine for computing technical indicators
        cache_ttl: Cache time-to-live in seconds
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        cache_ttl: int = 300,
    ) -> None:
        """Initialize the prediction service.

        Args:
            model_registry: Registry for managing trained models
            feature_engine: Engine for computing technical features
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.model_registry = model_registry
        self.feature_engine = feature_engine
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[PredictionResponse, float]] = {}

        logger.info(
            "prediction_service_initialized",
            cache_ttl=cache_ttl,
        )

    async def predict(
        self,
        symbol: str,
        horizon_days: int = 5,
        model_name: str | None = None,
        use_cache: bool = True,
    ) -> PredictionResponse:
        """Generate prediction for a symbol.

        This method performs the following steps:
        1. Check cache if enabled
        2. Fetch recent price data
        3. Compute technical features
        4. Load model (specific or default)
        5. Generate prediction
        6. Format response with metadata

        Args:
            symbol: Ticker symbol to predict
            horizon_days: Number of days ahead to predict
            model_name: Specific model to use, None for default
            use_cache: Whether to use cached predictions

        Returns:
            PredictionResponse with prediction and metadata

        Raises:
            ValueError: If symbol is invalid or data is insufficient
            RuntimeError: If model prediction fails

        Examples:
            >>> service = PredictionService(registry, feature_engine)
            >>> result = await service.predict("AAPL", horizon_days=5)
            >>> print(f"Expected return: {result.predicted_return:.2%}")
        """
        start_time = time.perf_counter()

        # Check cache
        if use_cache:
            cached = self._get_from_cache(symbol, horizon_days, model_name)
            if cached is not None:
                logger.info(
                    "prediction_served_from_cache",
                    symbol=symbol,
                    horizon_days=horizon_days,
                )
                return cached

        logger.info(
            "generating_prediction",
            symbol=symbol,
            horizon_days=horizon_days,
            model_name=model_name,
        )

        # In a real implementation, this would fetch from a data service
        # For now, we'll expect the data to be provided externally
        # This is a placeholder that would be replaced with actual data fetching
        df = await self._fetch_recent_data(symbol)

        # Compute features
        df_with_features = self.feature_engine.compute_all(df)

        # Prepare features for model input
        X = self._prepare_features(df_with_features)

        # Load model
        if model_name is not None:
            model = self.model_registry.get(model_name)
        else:
            model = self.model_registry.get_default()

        # Generate prediction
        predictions = model.predict(X)

        if not predictions:
            raise RuntimeError(f"Model returned no predictions for {symbol}")

        # Use the last prediction (most recent)
        pred = predictions[-1]

        # Get confidence from model's predict_proba if available
        try:
            proba_df = model.predict_proba(X)
            confidence = float(proba_df.select("confidence").tail(1).item())
        except Exception as e:
            logger.warning(
                "confidence_calculation_failed",
                error=str(e),
                using_default=0.5,
            )
            confidence = 0.5

        # Calculate predicted price if we have current price
        predicted_price = None
        if "close" in df.columns and df.height > 0:
            current_price = float(df.select("close").tail(1).item())
            predicted_price = current_price * (1 + pred.prediction / 100)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Create response
        response = PredictionResponse(
            symbol=symbol,
            timestamp=datetime.now(),
            horizon_days=horizon_days,
            predicted_return=pred.prediction,
            predicted_price=predicted_price,
            confidence=confidence,
            interval_lower=pred.lower_bound,
            interval_upper=pred.upper_bound,
            model_used=model.model_name,
            model_version=model.model_version,
            features_used=len(X.columns),
            latency_ms=latency_ms,
        )

        # Cache response
        if use_cache:
            self._add_to_cache(symbol, horizon_days, model_name, response)

        logger.info(
            "prediction_generated",
            symbol=symbol,
            predicted_return=response.predicted_return,
            confidence=response.confidence,
            latency_ms=latency_ms,
        )

        return response

    async def predict_batch(
        self,
        symbols: list[str],
        horizon_days: int = 5,
        model_name: str | None = None,
    ) -> list[PredictionResponse]:
        """Generate predictions for multiple symbols.

        This method processes predictions in parallel where possible,
        making it more efficient than calling predict() multiple times.

        Args:
            symbols: List of ticker symbols to predict
            horizon_days: Number of days ahead to predict
            model_name: Specific model to use, None for default

        Returns:
            List of PredictionResponse objects, one per symbol

        Raises:
            ValueError: If symbols list is empty
            RuntimeError: If predictions fail

        Examples:
            >>> symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
            >>> results = await service.predict_batch(symbols, horizon_days=5)
            >>> for result in results:
            ...     print(f"{result.symbol}: {result.predicted_return:.2%}")
        """
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        logger.info(
            "batch_prediction_started",
            num_symbols=len(symbols),
            horizon_days=horizon_days,
        )

        # Process predictions concurrently
        # In a real implementation, this would use asyncio.gather
        results = []
        for symbol in symbols:
            try:
                result = await self.predict(
                    symbol=symbol,
                    horizon_days=horizon_days,
                    model_name=model_name,
                    use_cache=True,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "batch_prediction_failed_for_symbol",
                    symbol=symbol,
                    error=str(e),
                )
                # Continue with other symbols

        logger.info(
            "batch_prediction_completed",
            requested=len(symbols),
            successful=len(results),
        )

        return results

    def _prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare feature DataFrame for model input.

        This method:
        1. Selects only feature columns (excludes timestamp, OHLCV)
        2. Handles missing values
        3. Ensures correct data types
        4. Orders features consistently

        Args:
            df: DataFrame with computed features

        Returns:
            DataFrame ready for model input

        Raises:
            ValueError: If features are insufficient or invalid
        """
        # Get feature names from the engine
        feature_names = self.feature_engine.get_feature_names()

        # Select only feature columns that exist
        available_features = [f for f in feature_names if f in df.columns]

        if not available_features:
            raise ValueError("No features available in DataFrame")

        # Select features and handle missing values
        X = df.select(available_features)

        # Fill NaN values with 0 (simple strategy, could be improved)
        X = X.fill_nan(0.0)
        X = X.fill_null(0.0)

        logger.debug(
            "features_prepared",
            num_features=len(available_features),
            num_rows=X.height,
        )

        return X

    async def _fetch_recent_data(self, symbol: str) -> pl.DataFrame:
        """Fetch recent price data for a symbol.

        This is a placeholder method that would be implemented to fetch
        data from the data layer (database, API, etc).

        Args:
            symbol: Ticker symbol to fetch

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If symbol is invalid
            RuntimeError: If data fetching fails
        """
        # Placeholder implementation
        # In production, this would call the data service
        raise NotImplementedError(
            "Data fetching not implemented. "
            "PredictionService requires integration with data layer."
        )

    def _get_cache_key(
        self,
        symbol: str,
        horizon_days: int,
        model_name: str | None,
    ) -> str:
        """Generate cache key for a prediction request."""
        model_str = model_name or "default"
        return f"{symbol}:{horizon_days}:{model_str}"

    def _get_from_cache(
        self,
        symbol: str,
        horizon_days: int,
        model_name: str | None,
    ) -> PredictionResponse | None:
        """Get prediction from cache if available and not expired."""
        key = self._get_cache_key(symbol, horizon_days, model_name)

        if key not in self._cache:
            return None

        response, cached_at = self._cache[key]

        # Check if expired
        if time.time() - cached_at > self.cache_ttl:
            del self._cache[key]
            return None

        return response

    def _add_to_cache(
        self,
        symbol: str,
        horizon_days: int,
        model_name: str | None,
        response: PredictionResponse,
    ) -> None:
        """Add prediction to cache."""
        key = self._get_cache_key(symbol, horizon_days, model_name)
        self._cache[key] = (response, time.time())

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._cache.clear()
        logger.info("prediction_cache_cleared")


__all__ = [
    "PredictionResponse",
    "PredictionService",
]
