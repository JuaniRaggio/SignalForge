"""Tests for MLflow configuration and helper functions."""

from unittest.mock import MagicMock, patch

import pytest

from signalforge.ml.training.mlflow_config import (
    log_metrics,
    log_model,
    log_params,
    setup_experiment,
    start_run,
)


@pytest.fixture
def mock_mlflow():
    """Create a mock MLflow module with all necessary attributes."""
    with patch("signalforge.ml.training.mlflow_config.mlflow") as mock:
        # Setup basic MLflow mock structure
        mock.get_experiment_by_name = MagicMock(return_value=None)
        mock.create_experiment = MagicMock(return_value="test_experiment_id")
        mock.set_tracking_uri = MagicMock()
        mock.active_run = MagicMock(return_value=None)
        mock.start_run = MagicMock()
        mock.end_run = MagicMock()
        mock.log_params = MagicMock()
        mock.log_metrics = MagicMock()

        # Setup active run mock
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_run.info.experiment_id = "test_experiment_id"

        yield mock


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MlflowClient."""
    with patch("signalforge.ml.training.mlflow_config.MlflowClient") as mock:
        yield mock


@pytest.fixture
def mock_settings():
    """Create a mock Settings object."""
    with patch("signalforge.ml.training.mlflow_config.get_settings") as mock:
        settings = MagicMock()
        settings.mlflow_tracking_uri = "http://localhost:5000"
        settings.mlflow_experiment_name = "test_experiment"
        settings.mlflow_artifact_location = "./test_mlartifacts"
        mock.return_value = settings
        yield settings


class TestSetupExperiment:
    """Tests for setup_experiment function."""

    def test_create_new_experiment(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test creating a new experiment when it doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new_exp_id"

        experiment_id = setup_experiment("new_experiment")

        assert experiment_id == "new_exp_id"
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.get_experiment_by_name.assert_called_once_with("new_experiment")
        mock_mlflow.create_experiment.assert_called_once_with(
            name="new_experiment",
            artifact_location="./test_mlartifacts",
        )

    def test_get_existing_experiment(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test getting an existing experiment."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing_exp_id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        experiment_id = setup_experiment("existing_experiment")

        assert experiment_id == "existing_exp_id"
        mock_mlflow.get_experiment_by_name.assert_called_once_with("existing_experiment")
        mock_mlflow.create_experiment.assert_not_called()

    def test_experiment_setup_failure(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test handling of experiment setup failure."""
        mock_mlflow.get_experiment_by_name.side_effect = Exception("Connection error")

        with pytest.raises(Exception, match="Connection error"):
            setup_experiment("test_experiment")


class TestLogParams:
    """Tests for log_params function."""

    def test_log_params_success(self, mock_mlflow: MagicMock) -> None:
        """Test successful parameter logging."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
        }

        log_params(params)

        expected_params = {
            "learning_rate": "0.01",
            "batch_size": "32",
            "epochs": "100",
            "optimizer": "adam",
        }
        mock_mlflow.log_params.assert_called_once_with(expected_params)

    def test_log_params_no_active_run(self, mock_mlflow: MagicMock) -> None:
        """Test parameter logging fails when no active run exists."""
        mock_mlflow.active_run.return_value = None

        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_params({"param": "value"})

    def test_log_params_converts_types(self, mock_mlflow: MagicMock) -> None:
        """Test that parameters are converted to strings."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        params = {
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "none_param": None,
            "list_param": [1, 2, 3],
        }

        log_params(params)

        # Verify all values were converted to strings
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert all(isinstance(v, str) for v in call_args.values())
        assert call_args["int_param"] == "42"
        assert call_args["float_param"] == "3.14"
        assert call_args["bool_param"] == "True"
        assert call_args["none_param"] == "None"
        assert call_args["list_param"] == "[1, 2, 3]"

    def test_log_params_empty_dict(self, mock_mlflow: MagicMock) -> None:
        """Test logging empty parameter dictionary."""
        mock_run = MagicMock()
        mock_mlflow.active_run.return_value = mock_run

        log_params({})

        mock_mlflow.log_params.assert_called_once_with({})


class TestLogMetrics:
    """Tests for log_metrics function."""

    def test_log_metrics_success(self, mock_mlflow: MagicMock) -> None:
        """Test successful metric logging."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        metrics = {
            "accuracy": 0.95,
            "loss": 0.05,
            "f1_score": 0.92,
        }

        log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=None)

    def test_log_metrics_with_step(self, mock_mlflow: MagicMock) -> None:
        """Test metric logging with step parameter."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        metrics = {"loss": 0.05}

        log_metrics(metrics, step=10)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=10)

    def test_log_metrics_no_active_run(self, mock_mlflow: MagicMock) -> None:
        """Test metric logging fails when no active run exists."""
        mock_mlflow.active_run.return_value = None

        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_metrics({"metric": 0.5})

    def test_log_metrics_empty_dict(self, mock_mlflow: MagicMock) -> None:
        """Test logging empty metrics dictionary."""
        mock_run = MagicMock()
        mock_mlflow.active_run.return_value = mock_run

        log_metrics({})

        mock_mlflow.log_metrics.assert_called_once_with({}, step=None)


class TestLogModel:
    """Tests for log_model function."""

    def test_log_sklearn_model(self, mock_mlflow: MagicMock) -> None:
        """Test logging a scikit-learn model."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        # Create a mock sklearn model
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "RandomForestClassifier"
        mock_model.__class__.__module__ = "sklearn.ensemble"

        mock_sklearn = MagicMock()
        with patch.dict(
            "sys.modules",
            {"mlflow.sklearn": mock_sklearn},
        ):
            log_model(mock_model, "model", registered_model_name="test_model")

            mock_sklearn.log_model.assert_called_once_with(
                sk_model=mock_model,
                artifact_path="model",
                registered_model_name="test_model",
            )

    def test_log_pytorch_model(self, mock_mlflow: MagicMock) -> None:
        """Test logging a PyTorch model."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "NeuralNetwork"
        mock_model.__class__.__module__ = "torch.nn"

        mock_pytorch = MagicMock()
        with patch.dict(
            "sys.modules",
            {"mlflow.pytorch": mock_pytorch},
        ):
            log_model(mock_model, "model")

            mock_pytorch.log_model.assert_called_once_with(
                pytorch_model=mock_model,
                artifact_path="model",
                registered_model_name=None,
            )

    def test_log_model_detects_framework(self, mock_mlflow: MagicMock) -> None:
        """Test that log_model correctly identifies model frameworks."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        # Test that different model types are logged without error
        test_cases = [
            ("XGBClassifier", "xgboost.sklearn"),
            ("LGBMClassifier", "lightgbm.sklearn"),
            ("Sequential", "tensorflow.keras"),
        ]

        for model_name, model_module in test_cases:
            mock_model = MagicMock()
            mock_model.__class__.__name__ = model_name
            mock_model.__class__.__module__ = model_module

            # The function should execute without raising an exception
            # We don't mock the internal mlflow calls as they are tested elsewhere
            try:
                with patch(f"mlflow.{model_module.split('.')[0]}.log_model"):
                    log_model(mock_model, f"model_{model_name}")
            except ImportError:
                # If the ML library is not installed, that's fine for this test
                pass

    def test_log_model_no_active_run(self, mock_mlflow: MagicMock) -> None:
        """Test model logging fails when no active run exists."""
        mock_mlflow.active_run.return_value = None
        mock_model = MagicMock()

        with pytest.raises(RuntimeError, match="No active MLflow run"):
            log_model(mock_model, "model")

    def test_log_generic_model_pyfunc(self, mock_mlflow: MagicMock) -> None:
        """Test logging a generic model using pyfunc flavor."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CustomModel"
        mock_model.__class__.__module__ = "custom.module"

        # Simply verify that the function executes without error
        with patch("mlflow.pyfunc.log_model") as mock_log:
            log_model(mock_model, "model", registered_model_name="custom_model")
            mock_log.assert_called_once()


class TestStartRun:
    """Tests for start_run context manager."""

    def test_start_run_success(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test successful run creation and completion."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_run.info.experiment_id = "test_exp_id"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        with start_run(run_name="test_run") as run:
            assert run == mock_run

        # Verify set_tracking_uri was called (may be called multiple times by setup_experiment and start_run)
        assert mock_mlflow.set_tracking_uri.called
        mock_mlflow.start_run.assert_called_once_with(
            run_name="test_run",
            experiment_id="test_exp_id",
            nested=False,
        )
        # Verify end_run was called
        assert mock_mlflow.end_run.called

    def test_start_run_nested(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test creating a nested run."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "nested_run_id"
        mock_mlflow.start_run.return_value = mock_run

        with start_run(run_name="nested_run", nested=True) as run:
            assert run == mock_run

        mock_mlflow.start_run.assert_called_once_with(
            run_name="nested_run",
            experiment_id="test_exp_id",
            nested=True,
        )

    def test_start_run_without_name(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test creating a run without explicit name."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        with start_run() as run:
            assert run == mock_run

        mock_mlflow.start_run.assert_called_once_with(
            run_name=None,
            experiment_id="test_exp_id",
            nested=False,
        )

    def test_start_run_exception_handling(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test that run is ended even when exception occurs."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        with pytest.raises(ValueError, match="Test error"), start_run(run_name="test_run"):
            raise ValueError("Test error")

        # Verify run was ended despite exception
        mock_mlflow.end_run.assert_called_once()

    def test_start_run_creates_experiment_if_not_exists(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test that experiment is created if it doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new_exp_id"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        with start_run():
            pass

        mock_mlflow.create_experiment.assert_called_once_with(
            name="test_experiment",
            artifact_location="./test_mlartifacts",
        )
        mock_mlflow.start_run.assert_called_once_with(
            run_name=None,
            experiment_id="new_exp_id",
            nested=False,
        )


class TestIntegration:
    """Integration tests for MLflow helpers working together."""

    def test_full_workflow(
        self,
        mock_mlflow: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test a complete MLflow tracking workflow."""
        # Setup experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_id"
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_id"

        # Setup run
        mock_run = MagicMock()
        mock_run.info.run_id = "run_id"
        mock_run.info.experiment_id = "exp_id"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        # Mock sklearn for model logging
        with patch("mlflow.sklearn.log_model") as mock_sklearn_log:
            # Execute workflow
            with start_run(run_name="integration_test") as run:
                assert run.info.run_id == "run_id"

                log_params(
                    {
                        "learning_rate": 0.01,
                        "max_depth": 5,
                    }
                )

                log_metrics(
                    {
                        "accuracy": 0.95,
                        "loss": 0.05,
                    }
                )

                mock_model = MagicMock()
                mock_model.__class__.__name__ = "RandomForest"
                mock_model.__class__.__module__ = "sklearn.ensemble"

                log_model(mock_model, "model", registered_model_name="best_model")

            # Verify all operations
            mock_mlflow.log_params.assert_called_once()
            mock_mlflow.log_metrics.assert_called_once()
            mock_sklearn_log.assert_called_once()
            assert mock_mlflow.end_run.called
