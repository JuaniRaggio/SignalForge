"""Tests for Machine Learning API endpoints."""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_predict_success(client: AsyncClient) -> None:
    """Test successful prediction request."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlpredict@example.com",
            "username": "mluser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlpredict@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Request prediction
    response = await client.post(
        "/api/v1/ml/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "horizon": 5,
            "include_features": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "current_price" in data
    assert "predictions" in data
    assert len(data["predictions"]) == 5
    assert data["model_id"] is not None
    assert data["model_name"] is not None

    # Verify prediction structure
    prediction = data["predictions"][0]
    assert "horizon" in prediction
    assert "predicted_price" in prediction
    assert "predicted_direction" in prediction
    assert "confidence" in prediction
    assert 0.0 <= prediction["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_predict_invalid_symbol(client: AsyncClient) -> None:
    """Test prediction with invalid symbol."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlinvalid@example.com",
            "username": "mlinvaliduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlinvalid@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/ml/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "INVALID",
            "horizon": 1,
        },
    )

    assert response.status_code == 404
    # Exception handler uses nested format: {"error": {"message": "..."}}
    data = response.json()
    msg = data.get("error", {}).get("message", data.get("detail", "")).lower()
    assert "not found" in msg or "no prediction" in msg or "invalid" in msg


@pytest.mark.asyncio
async def test_predict_unauthorized(client: AsyncClient) -> None:
    """Test prediction without authentication."""
    response = await client.post(
        "/api/v1/ml/predict",
        json={
            "symbol": "AAPL",
            "horizon": 1,
        },
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_predict_with_model_id(client: AsyncClient) -> None:
    """Test prediction with specific model ID."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlmodelid@example.com",
            "username": "mlmodeliduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlmodelid@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/ml/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "MSFT",
            "horizon": 3,
            "model_id": "ensemble_v1",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "ensemble_v1"


@pytest.mark.asyncio
async def test_predict_batch_success(client: AsyncClient) -> None:
    """Test successful batch prediction."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlbatch@example.com",
            "username": "mlbatchuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlbatch@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/ml/predict/batch",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "horizon": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert all(item["symbol"] in ["AAPL", "MSFT", "GOOGL"] for item in data)
    assert all(len(item["predictions"]) == 3 for item in data)


@pytest.mark.asyncio
async def test_predict_batch_empty_symbols(client: AsyncClient) -> None:
    """Test batch prediction with empty symbol list."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlbatchempty@example.com",
            "username": "mlbatchemptyuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlbatchempty@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/ml/predict/batch",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbols": [],
            "horizon": 1,
        },
    )

    assert response.status_code == 400  # Validation error


@pytest.mark.asyncio
async def test_list_models_success(client: AsyncClient) -> None:
    """Test listing available models."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlmodels@example.com",
            "username": "mlmodelsuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlmodels@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/ml/models",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data
    assert isinstance(data["models"], list)
    assert data["count"] == len(data["models"])


@pytest.mark.asyncio
async def test_list_models_with_filter(client: AsyncClient) -> None:
    """Test listing models with type filter."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlfilter@example.com",
            "username": "mlfilteruser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlfilter@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/ml/models?model_type=ensemble",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert all(model["model_type"] == "ensemble" for model in data["models"])


@pytest.mark.asyncio
async def test_list_models_active_only(client: AsyncClient) -> None:
    """Test listing only active models."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlactive@example.com",
            "username": "mlactiveuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlactive@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/ml/models?is_active=true",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert all(model["is_active"] for model in data["models"])


@pytest.mark.asyncio
async def test_get_model_success(client: AsyncClient) -> None:
    """Test getting model details."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mldetails@example.com",
            "username": "mldetailsuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mldetails@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/ml/models/ensemble_v1",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "ensemble_v1"
    assert "name" in data
    assert "model_type" in data
    assert "version" in data
    assert "metrics" in data
    assert "hyperparameters" in data
    assert "is_active" in data
    assert "training_period" in data
    assert "feature_count" in data


@pytest.mark.asyncio
async def test_get_model_not_found(client: AsyncClient) -> None:
    """Test getting non-existent model."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlnotfound@example.com",
            "username": "mlnotfounduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlnotfound@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/ml/models/nonexistent",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_backtest_success(client: AsyncClient) -> None:
    """Test successful backtest."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlbacktest@example.com",
            "username": "mlbacktestuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlbacktest@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    start_date = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    end_date = (datetime.now(UTC) - timedelta(days=30)).isoformat()

    response = await client.post(
        "/api/v1/ml/backtest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "model_id": "ensemble_v1",
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": "100000.00",
            "position_size": "0.10",
            "transaction_cost": "0.001",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["model_id"] == "ensemble_v1"
    assert "metrics" in data
    assert "trades" in data
    assert "equity_curve" in data

    # Verify metrics structure
    metrics = data["metrics"]
    assert "total_return" in metrics
    assert "annualized_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics
    assert "total_trades" in metrics
    assert "winning_trades" in metrics
    assert "losing_trades" in metrics


@pytest.mark.asyncio
async def test_backtest_invalid_model(client: AsyncClient) -> None:
    """Test backtest with invalid model."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlbacktestinvalid@example.com",
            "username": "mlbacktestinvaliduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlbacktestinvalid@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    start_date = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    end_date = (datetime.now(UTC) - timedelta(days=30)).isoformat()

    response = await client.post(
        "/api/v1/ml/backtest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "model_id": "invalid_model",
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_backtest_invalid_dates(client: AsyncClient) -> None:
    """Test backtest with end date before start date."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlbacktestdates@example.com",
            "username": "mlbacktestdatesuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlbacktestdates@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # End date before start date
    start_date = datetime.now(UTC).isoformat()
    end_date = (datetime.now(UTC) - timedelta(days=365)).isoformat()

    response = await client.post(
        "/api/v1/ml/backtest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "model_id": "ensemble_v1",
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    # Should accept but may return validation error or empty results
    # Depending on implementation
    assert response.status_code in [200, 400, 422]


@pytest.mark.asyncio
async def test_predict_validation_horizon_too_large(client: AsyncClient) -> None:
    """Test prediction with horizon exceeding maximum."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlhorizon@example.com",
            "username": "mlhorizonuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlhorizon@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/ml/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "horizon": 100,  # Exceeds max of 30
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_predict_validation_horizon_zero(client: AsyncClient) -> None:
    """Test prediction with zero horizon."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlhorizonzero@example.com",
            "username": "mlhorizonzerouser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlhorizonzero@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/ml/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "horizon": 0,
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_batch_predict_too_many_symbols(client: AsyncClient) -> None:
    """Test batch prediction with too many symbols."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlbatchlarge@example.com",
            "username": "mlbatchlargeuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlbatchlarge@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create list with more than 100 symbols
    symbols = [f"SYM{i:03d}" for i in range(150)]

    response = await client.post(
        "/api/v1/ml/predict/batch",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbols": symbols,
            "horizon": 1,
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_backtest_custom_parameters(client: AsyncClient) -> None:
    """Test backtest with custom parameters."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mlbacktestcustom@example.com",
            "username": "mlbacktestcustomuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mlbacktestcustom@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    start_date = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    end_date = (datetime.now(UTC) - timedelta(days=30)).isoformat()

    response = await client.post(
        "/api/v1/ml/backtest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "model_id": "ensemble_v1",
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": "50000.00",
            "position_size": "0.05",
            "transaction_cost": "0.002",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["initial_capital"] == "50000.00"


@pytest.mark.asyncio
async def test_predict_case_insensitive_symbol(client: AsyncClient) -> None:
    """Test prediction with lowercase symbol."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mllowercase@example.com",
            "username": "mllowercaseuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mllowercase@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/ml/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "aapl",  # lowercase
            "horizon": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"  # Should be converted to uppercase
