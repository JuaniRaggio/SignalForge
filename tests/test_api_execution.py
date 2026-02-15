"""Tests for Execution Quality API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_liquidity_score_success(client: AsyncClient) -> None:
    """Test successful liquidity score retrieval."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execliquidity@example.com",
            "username": "execliquidityuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execliquidity@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/execution/liquidity/AAPL",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "liquidity_score" in data
    assert 0.0 <= data["liquidity_score"] <= 100.0
    assert data["liquidity_tier"] in ["high", "medium", "low", "very_low"]
    assert "metrics" in data
    assert "recommendation" in data


@pytest.mark.asyncio
async def test_get_liquidity_score_invalid_symbol(client: AsyncClient) -> None:
    """Test liquidity score with invalid symbol."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execliquidityinvalid@example.com",
            "username": "execliquidityinvaliduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execliquidityinvalid@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/execution/liquidity/INVALID",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_estimate_slippage_success(client: AsyncClient) -> None:
    """Test successful slippage estimation."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execslippage@example.com",
            "username": "execslippageuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execslippage@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/execution/slippage/estimate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "order_size": 1000,
            "side": "buy",
            "urgency": "normal",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["order_size"] == 1000
    assert data["side"] == "buy"
    assert "current_price" in data
    assert "estimated_execution_price" in data
    assert "slippage" in data
    assert "estimated_cost" in data
    assert 0.0 <= data["confidence"] <= 1.0

    # Verify slippage components
    slippage = data["slippage"]
    assert "market_impact_bps" in slippage
    assert "spread_cost_bps" in slippage
    assert "timing_risk_bps" in slippage
    assert "total_slippage_bps" in slippage


@pytest.mark.asyncio
async def test_estimate_slippage_sell_side(client: AsyncClient) -> None:
    """Test slippage estimation for sell order."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execslippagesell@example.com",
            "username": "execslippageselluser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execslippagesell@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/execution/slippage/estimate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "MSFT",
            "order_size": 500,
            "side": "sell",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["side"] == "sell"


@pytest.mark.asyncio
async def test_estimate_slippage_high_urgency(client: AsyncClient) -> None:
    """Test slippage estimation with high urgency."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execslippageurgent@example.com",
            "username": "execslippageurgentuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execslippageurgent@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/execution/slippage/estimate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "GOOGL",
            "order_size": 1000,
            "side": "buy",
            "urgency": "high",
        },
    )

    assert response.status_code == 200
    # High urgency should typically result in higher slippage
    data = response.json()
    assert data["slippage"]["timing_risk_bps"] is not None


@pytest.mark.asyncio
async def test_estimate_slippage_invalid_side(client: AsyncClient) -> None:
    """Test slippage estimation with invalid side."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execslippageinvalidside@example.com",
            "username": "execslippageinvalidsideuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execslippageinvalidside@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/execution/slippage/estimate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "order_size": 1000,
            "side": "invalid",
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_spread_metrics_success(client: AsyncClient) -> None:
    """Test successful spread metrics retrieval."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execspread@example.com",
            "username": "execspreaduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execspread@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/execution/spread/AAPL",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "metrics" in data
    assert "is_favorable" in data
    assert "recommendation" in data

    # Verify metrics structure
    metrics = data["metrics"]
    assert "current_bid" in metrics
    assert "current_ask" in metrics
    assert "spread_absolute" in metrics
    assert "spread_bps" in metrics
    assert "avg_spread_1h" in metrics
    assert "avg_spread_1d" in metrics
    assert "spread_percentile" in metrics
    assert 0.0 <= metrics["spread_percentile"] <= 100.0


@pytest.mark.asyncio
async def test_check_volume_filter_success(client: AsyncClient) -> None:
    """Test successful volume filter check."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execvolume@example.com",
            "username": "execvolumeuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execvolume@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/execution/volume-filter",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "order_size": 100000,
            "max_volume_participation": "0.05",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["order_size"] == 100000
    assert "passes_filter" in data
    assert "analysis" in data
    assert "warnings" in data
    assert "recommendation" in data

    # Verify analysis structure
    analysis = data["analysis"]
    assert "current_volume" in analysis
    assert "avg_volume_20d" in analysis
    assert "order_as_pct_avg_volume" in analysis
    assert "estimated_time_to_fill_minutes" in analysis
    assert "volume_profile" in analysis


@pytest.mark.asyncio
async def test_check_volume_filter_large_order(client: AsyncClient) -> None:
    """Test volume filter with large order."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execvolumelarge@example.com",
            "username": "execvolumelargeuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execvolumelarge@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/execution/volume-filter",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "order_size": 10000000,  # Very large order
        },
    )

    assert response.status_code == 200
    data = response.json()
    # Large order likely won't pass filter
    assert isinstance(data["passes_filter"], bool)
    if not data["passes_filter"]:
        assert len(data["warnings"]) > 0


@pytest.mark.asyncio
async def test_get_execution_quality_success(client: AsyncClient) -> None:
    """Test comprehensive execution quality assessment."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execquality@example.com",
            "username": "execqualityuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execquality@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/execution/execution-quality/AAPL?order_size=1000&side=buy",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["order_size"] == 1000
    assert data["side"] == "buy"
    assert "current_price" in data
    assert "metrics" in data
    assert "liquidity" in data
    assert "slippage" in data
    assert "spread" in data
    assert "volume" in data
    assert "overall_recommendation" in data
    assert "warnings" in data
    assert "is_tradeable" in data

    # Verify metrics structure
    metrics = data["metrics"]
    assert "liquidity_score" in metrics
    assert "estimated_slippage_bps" in metrics
    assert "spread_bps" in metrics
    assert "volume_participation_pct" in metrics
    assert "execution_difficulty" in metrics
    assert metrics["execution_difficulty"] in ["easy", "moderate", "difficult", "very_difficult"]
    assert "overall_score" in metrics
    assert 0.0 <= metrics["overall_score"] <= 100.0


@pytest.mark.asyncio
async def test_get_execution_quality_missing_params(client: AsyncClient) -> None:
    """Test execution quality without required parameters."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execqualitymissing@example.com",
            "username": "execqualitymissinguser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execqualitymissing@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Missing order_size and side
    response = await client.get(
        "/api/v1/execution/execution-quality/AAPL",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_liquidity_score_case_insensitive(client: AsyncClient) -> None:
    """Test liquidity score with lowercase symbol."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execliquiditylower@example.com",
            "username": "execliquidityloweruser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execliquiditylower@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/execution/liquidity/aapl",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_slippage_unauthorized(client: AsyncClient) -> None:
    """Test slippage estimation without authentication."""
    response = await client.post(
        "/api/v1/execution/slippage/estimate",
        json={
            "symbol": "AAPL",
            "order_size": 1000,
            "side": "buy",
        },
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_volume_filter_zero_order_size(client: AsyncClient) -> None:
    """Test volume filter with zero order size."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execvolumezero@example.com",
            "username": "execvolumezerouser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execvolumezero@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.post(
        "/api/v1/execution/volume-filter",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "symbol": "AAPL",
            "order_size": 0,
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_execution_quality_sell_side(client: AsyncClient) -> None:
    """Test execution quality for sell order."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "execqualitysell@example.com",
            "username": "execqualityselluser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "execqualitysell@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/execution/execution-quality/MSFT?order_size=500&side=sell",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["side"] == "sell"
