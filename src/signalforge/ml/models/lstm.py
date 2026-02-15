"""LSTM deep learning model for time series prediction.

This module provides PyTorch-based LSTM neural networks for financial time
series forecasting. The implementation includes proper sequence handling,
normalization, early stopping, and model persistence.

The LSTMPredictor class follows the BasePredictor interface and provides
enterprise-grade features like device auto-detection (CUDA/MPS/CPU),
gradient clipping, and ONNX export for production deployment.

Examples:
    Basic usage:

    >>> import polars as pl
    >>> from signalforge.ml.models.lstm import LSTMPredictor
    >>>
    >>> df = pl.DataFrame({
    ...     "timestamp": pl.date_range(start="2024-01-01", periods=100, interval="1d"),
    ...     "close": [100.0 + i * 0.5 for i in range(100)],
    ...     "volume": [1000000] * 100,
    ... })
    >>> model = LSTMPredictor(input_size=2, sequence_length=20)
    >>> X = df.select(["close", "volume"])
    >>> y = df.select("close").to_series()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X.tail(20))

    Custom configuration:

    >>> model = LSTMPredictor(
    ...     input_size=5,
    ...     sequence_length=30,
    ...     hidden_size=128,
    ...     num_layers=3,
    ...     dropout=0.3,
    ...     learning_rate=0.0005,
    ...     batch_size=64,
    ...     epochs=200,
    ...     early_stopping_patience=15,
    ... )
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from signalforge.ml.models.base import BasePredictor, PredictionResult

if TYPE_CHECKING:
    import polars as pl

logger = structlog.get_logger(__name__)


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network architecture.

    This module implements a multi-layer LSTM with dropout regularization
    for time series prediction. The network consists of stacked LSTM layers
    followed by a fully connected output layer.

    Attributes:
        input_size: Number of input features per time step.
        hidden_size: Number of features in hidden state.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability for regularization.
        output_size: Dimensionality of output (default: 1 for regression).
        lstm: LSTM layers.
        fc: Fully connected output layer.

    Examples:
        >>> network = LSTMNetwork(input_size=5, hidden_size=64, num_layers=2)
        >>> x = torch.randn(32, 20, 5)  # batch_size=32, seq_len=20, features=5
        >>> output = network(x)
        >>> output.shape
        torch.Size([32, 1])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """Initialize LSTM network.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units in LSTM layers.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers (0 if num_layers=1).
            output_size: Number of output features (typically 1 for regression).

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers
        # Note: dropout is only applied if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        # LSTM forward pass
        # lstm_out: (batch_size, seq_len, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        # c_n: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last time step output
        # last_output: (batch_size, hidden_size)
        last_output: torch.Tensor = lstm_out[:, -1, :]

        # Fully connected layer
        # output: (batch_size, output_size)
        output: torch.Tensor = self.fc(last_output)

        return output


class LSTMPredictor(BasePredictor):
    """LSTM model for time series prediction.

    This class provides a complete LSTM-based prediction pipeline including
    data preprocessing, sequence generation, model training with early stopping,
    and inference. It handles device management (CUDA/MPS/CPU), normalization,
    and provides model persistence capabilities.

    Attributes:
        model_name: Name identifier for the model.
        model_version: Version string for the model.
        input_size: Number of input features.
        sequence_length: Length of input sequences.
        hidden_size: LSTM hidden state size.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        early_stopping_patience: Patience for early stopping.
        device: Torch device (cuda/mps/cpu).
        network: LSTMNetwork instance.
        scaler_mean: Feature means for normalization.
        scaler_std: Feature standard deviations for normalization.
        is_fitted: Whether model has been trained.

    Examples:
        >>> predictor = LSTMPredictor(input_size=5, sequence_length=20)
        >>> predictor.fit(X_train, y_train)
        >>> predictions = predictor.predict(X_test)
        >>> predictor.save("/path/to/model.pkl")
        >>> loaded = LSTMPredictor.load("/path/to/model.pkl")
    """

    model_name = "lstm"
    model_version = "1.0.0"

    def __init__(
        self,
        input_size: int,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str | None = None,
    ):
        """Initialize LSTM predictor.

        Args:
            input_size: Number of input features.
            sequence_length: Number of time steps in input sequences.
            hidden_size: Number of hidden units in LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability for regularization.
            learning_rate: Learning rate for Adam optimizer.
            batch_size: Batch size for training.
            epochs: Maximum number of training epochs.
            early_stopping_patience: Epochs without improvement before stopping.
            device: Device to use ("cuda", "mps", "cpu"). Auto-detected if None.

        Raises:
            ValueError: If parameters are invalid.
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {early_stopping_patience}"
            )

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Initialize network
        self.network = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=1,
        ).to(self.device)

        # Normalization parameters (fitted during training)
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

        # Training state
        self.is_fitted = False

        logger.info(
            "Initialized LSTMPredictor",
            input_size=input_size,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=str(self.device),
        )

    def fit(self, X: pl.DataFrame, y: pl.Series, **_kwargs: Any) -> LSTMPredictor:
        """Train LSTM model.

        Performs the following steps:
        1. Normalizes features using z-score normalization
        2. Creates sequences of specified length
        3. Trains network with early stopping
        4. Tracks best model based on validation loss

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target series (n_samples,).
            **kwargs: Additional arguments (currently unused).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X and y have incompatible shapes or insufficient data.
            RuntimeError: If training fails.

        Examples:
            >>> predictor = LSTMPredictor(input_size=3)
            >>> predictor.fit(X_train, y_train)
            >>> # Model is now ready for prediction
        """

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: X={len(X)}, y={len(y)}")

        if len(X) < self.sequence_length + 1:
            raise ValueError(
                f"Insufficient data: need at least {self.sequence_length + 1} samples, "
                f"got {len(X)}"
            )

        if X.width != self.input_size:
            raise ValueError(
                f"X must have {self.input_size} features, got {X.width}"
            )

        logger.info("Starting LSTM training", samples=len(X), features=X.width)

        # Normalize features
        X_normalized, scaler_params = self._normalize(X)
        self.scaler_mean = scaler_params["mean"]
        self.scaler_std = scaler_params["std"]

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_normalized, y)

        if len(X_seq) == 0:
            raise ValueError("No sequences created. Check data length and sequence_length.")

        logger.info("Created sequences", n_sequences=len(X_seq))

        # Split into train/validation (80/20)
        n_train = int(len(X_seq) * 0.8)
        X_train = X_seq[:n_train]
        y_train = y_seq[:n_train]
        X_val = X_seq[n_train:]
        y_val = y_seq[n_train:]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict = None

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.network(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            self.network.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.network(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state_dict = self.network.state_dict().copy()
                logger.debug(
                    "New best model",
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                )
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Training progress",
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    patience=patience_counter,
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(
                    "Early stopping triggered",
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                )
                break

        # Restore best model
        if best_state_dict is not None:
            self.network.load_state_dict(best_state_dict)

        self.is_fitted = True
        logger.info("LSTM training complete", best_val_loss=best_val_loss)

        return self

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Generate predictions for the given input features.

        Args:
            X: Feature matrix for prediction.

        Returns:
            List of PredictionResult objects with predictions.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If X has incorrect number of features.

        Examples:
            >>> predictions = predictor.predict(X_test)
            >>> for pred in predictions:
            ...     print(f"Prediction: {pred.prediction:.2f}")
        """
        import polars as pl

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if X.width != self.input_size:
            raise ValueError(
                f"X must have {self.input_size} features, got {X.width}"
            )

        if len(X) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} samples for prediction, got {len(X)}"
            )

        # Normalize features
        if self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Scaler parameters not found. Model may not be properly fitted.")

        X_array = X.to_numpy()
        X_normalized = (X_array - self.scaler_mean) / (self.scaler_std + 1e-8)
        X_normalized_df = pl.DataFrame(X_normalized, schema=X.columns)

        # Create sequences
        # For prediction, we create overlapping sequences
        sequences = []
        for i in range(len(X_normalized_df) - self.sequence_length + 1):
            seq = X_normalized_df[i : i + self.sequence_length].to_numpy()
            sequences.append(seq)

        if not sequences:
            raise ValueError("Could not create sequences from input data")

        X_seq = torch.tensor(np.array(sequences), dtype=torch.float32).to(self.device)

        # Predict
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(X_seq)
            predictions_array = outputs.cpu().numpy().squeeze()

        # Create PredictionResult objects
        results = []
        for pred in predictions_array:
            result = PredictionResult(
                symbol="",  # Symbol not available in this context
                timestamp=datetime.now(),
                horizon_days=1,
                prediction=float(pred),
                confidence=0.0,  # LSTM doesn't have natural confidence, use ensemble for that
                model_name=self.model_name,
                model_version=self.model_version,
            )
            results.append(result)

        return results

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return predictions (LSTM doesn't have natural confidence).

        For regression LSTMs, there's no natural probability distribution.
        This method returns predictions as a DataFrame for interface consistency.
        Use ensemble methods for confidence intervals.

        Args:
            X: Feature matrix for prediction.

        Returns:
            DataFrame with predictions.

        Examples:
            >>> proba = predictor.predict_proba(X_test)
            >>> print(proba)
        """
        import polars as pl

        predictions = self.predict(X)
        return pl.DataFrame(
            {
                "prediction": [p.prediction for p in predictions],
                "confidence": [p.confidence for p in predictions],
            }
        )

    def _create_sequences(
        self, X: pl.DataFrame, y: pl.Series
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create input sequences for LSTM.

        Converts time series data into overlapping sequences of fixed length.
        Each sequence predicts the next value.

        Args:
            X: Feature DataFrame.
            y: Target series.

        Returns:
            Tuple of (X_sequences, y_targets) as torch tensors.

        Examples:
            >>> X_seq, y_seq = predictor._create_sequences(X, y)
            >>> X_seq.shape  # (n_sequences, sequence_length, n_features)
            >>> y_seq.shape  # (n_sequences,)
        """
        X_array = X.to_numpy()
        y_array = y.to_numpy()

        sequences = []
        targets = []

        for i in range(len(X_array) - self.sequence_length):
            # Input sequence
            seq = X_array[i : i + self.sequence_length]
            sequences.append(seq)

            # Target is the next value
            target = y_array[i + self.sequence_length]
            targets.append(target)

        X_seq = torch.tensor(np.array(sequences), dtype=torch.float32)
        y_seq = torch.tensor(np.array(targets), dtype=torch.float32)

        return X_seq, y_seq

    def _normalize(self, X: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
        """Normalize features using z-score normalization.

        Args:
            X: Feature DataFrame.

        Returns:
            Tuple of (normalized_df, scaler_params).
            scaler_params contains 'mean' and 'std' arrays.

        Examples:
            >>> X_norm, params = predictor._normalize(X)
            >>> # X_norm has mean ~0 and std ~1
        """
        import polars as pl

        X_array = X.to_numpy()

        # Calculate mean and std
        mean = np.mean(X_array, axis=0)
        std = np.std(X_array, axis=0)

        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)

        # Normalize
        X_normalized = (X_array - mean) / std

        # Convert back to DataFrame
        X_normalized_df = pl.DataFrame(X_normalized, schema=X.columns)

        scaler_params = {"mean": mean, "std": std}

        return X_normalized_df, scaler_params

    def to_onnx(self, path: str) -> None:
        """Export model to ONNX format for production deployment.

        Args:
            path: File path where ONNX model should be saved.

        Raises:
            RuntimeError: If model has not been fitted or export fails.

        Examples:
            >>> predictor.fit(X_train, y_train)
            >>> predictor.to_onnx("/path/to/model.onnx")
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before ONNX export")

        try:
            # Create dummy input
            dummy_input = torch.randn(
                1, self.sequence_length, self.input_size, device=self.device
            )

            # Export to ONNX
            torch.onnx.export(
                self.network,
                (dummy_input,),
                path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )

            logger.info("Exported model to ONNX", path=path)

        except Exception as e:
            logger.error("ONNX export failed", error=str(e))
            raise RuntimeError(f"ONNX export failed: {e}") from e

    def save(self, path: str) -> None:
        """Save model state dict and config.

        Saves both the network weights and all configuration parameters
        needed to reconstruct the model.

        Args:
            path: File path where model should be saved.

        Raises:
            IOError: If saving fails.

        Examples:
            >>> predictor.save("/path/to/model.pkl")
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "network_state_dict": self.network.state_dict(),
                "config": {
                    "input_size": self.input_size,
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "early_stopping_patience": self.early_stopping_patience,
                },
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "is_fitted": self.is_fitted,
                "model_name": self.model_name,
                "model_version": self.model_version,
            }

            with open(save_path, "wb") as f:
                pickle.dump(state, f)

            logger.info("Saved model", path=path)

        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise OSError(f"Failed to save model: {e}") from e

    @classmethod
    def load(cls, path: str) -> LSTMPredictor:
        """Load model from saved state.

        Reconstructs a fully trained LSTMPredictor from disk.

        Args:
            path: File path from which to load the model.

        Returns:
            Loaded LSTMPredictor instance.

        Raises:
            IOError: If loading fails.
            ValueError: If saved model is invalid.

        Examples:
            >>> loaded = LSTMPredictor.load("/path/to/model.pkl")
            >>> predictions = loaded.predict(X_test)
        """
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            # Create instance with saved config
            config = state["config"]
            instance = cls(
                input_size=config["input_size"],
                sequence_length=config["sequence_length"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                epochs=config["epochs"],
                early_stopping_patience=config["early_stopping_patience"],
            )

            # Load network state
            instance.network.load_state_dict(state["network_state_dict"])
            instance.network.to(instance.device)

            # Load scaler parameters
            instance.scaler_mean = state["scaler_mean"]
            instance.scaler_std = state["scaler_std"]
            instance.is_fitted = state["is_fitted"]

            logger.info("Loaded model", path=path)

            return instance

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise OSError(f"Failed to load model: {e}") from e

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances if available.

        LSTM models don't have built-in feature importance.
        Returns empty dict for interface consistency.

        Returns:
            Empty dictionary.

        Note:
            For feature importance analysis, use SHAP or similar methods.
        """
        return {}


__all__ = ["LSTMNetwork", "LSTMPredictor"]
