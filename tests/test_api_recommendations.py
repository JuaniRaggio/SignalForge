"""Tests for recommendation API endpoints."""

import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.fixture
async def auth_token(client: AsyncClient) -> str:
    """Create a test user and return auth token."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "recom@example.com",
            "username": "recomuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "recom@example.com",
            "password": "testpassword123",
        },
    )
    return login_response.json()["access_token"]


class TestGetPersonalizedFeed:
    """Tests for GET /feed endpoint."""

    @pytest.mark.asyncio
    async def test_get_feed_success(self, client: AsyncClient, auth_token: str) -> None:
        """Test successful feed retrieval."""
        response = await client.get(
            "/api/v1/recommendations/feed",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data
        assert "items" in data
        assert "generated_at" in data
        assert "next_refresh_at" in data

    @pytest.mark.asyncio
    async def test_get_feed_with_feed_type(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test feed retrieval with specific feed type."""
        response = await client.get(
            "/api/v1/recommendations/feed?feed_type=signals&limit=10",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_get_feed_unauthorized(self, client: AsyncClient) -> None:
        """Test feed retrieval without authentication."""
        response = await client.get("/api/v1/recommendations/feed")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestGetDailyDigest:
    """Tests for GET /digest/daily endpoint."""

    @pytest.mark.asyncio
    async def test_get_daily_digest_success(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test successful daily digest retrieval."""
        response = await client.get(
            "/api/v1/recommendations/digest/daily",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data
        assert "digest_date" in data
        assert "watchlist_signals" in data
        assert "portfolio_alerts" in data
        assert "sector_highlights" in data
        assert "upcoming_events" in data
        assert "summary_text" in data


class TestGetWeeklySummary:
    """Tests for GET /digest/weekly endpoint."""

    @pytest.mark.asyncio
    async def test_get_weekly_summary_success(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test successful weekly summary retrieval."""
        response = await client.get(
            "/api/v1/recommendations/digest/weekly",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data
        assert "week_start" in data
        assert "week_end" in data
        assert "portfolio_performance" in data
        assert "top_signals_accuracy" in data
        assert "engagement_stats" in data
        assert "recommendations_for_next_week" in data


class TestSubmitFeedback:
    """Tests for POST /feedback endpoint."""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test successful feedback submission."""
        feedback_data = {
            "item_id": "signal_001",
            "feedback_type": "like",
            "rating": 5,
            "comment": "Great signal!",
        }

        response = await client.post(
            "/api/v1/recommendations/feedback",
            json=feedback_data,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "message" in data

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_type(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test feedback submission with invalid type."""
        feedback_data = {
            "item_id": "signal_001",
            "feedback_type": "invalid_type",
        }

        response = await client.post(
            "/api/v1/recommendations/feedback",
            json=feedback_data,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestGetProfile:
    """Tests for GET /profile endpoint."""

    @pytest.mark.asyncio
    async def test_get_profile_success(self, client: AsyncClient, auth_token: str) -> None:
        """Test successful profile retrieval."""
        response = await client.get(
            "/api/v1/recommendations/profile",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data
        assert "risk_tolerance" in data
        assert "preferred_sectors" in data
        assert "watchlist" in data


class TestUpdateProfile:
    """Tests for PUT /profile endpoint."""

    @pytest.mark.asyncio
    async def test_update_profile_success(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test successful profile update."""
        update_data = {
            "risk_tolerance": "high",
            "preferred_sectors": ["Technology", "Finance"],
            "watchlist": ["AAPL", "MSFT", "GOOGL"],
        }

        response = await client.put(
            "/api/v1/recommendations/profile",
            json=update_data,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_update_profile_invalid_risk_tolerance(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test profile update with invalid risk tolerance."""
        update_data = {
            "risk_tolerance": "invalid",
        }

        response = await client.put(
            "/api/v1/recommendations/profile",
            json=update_data,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestOnboarding:
    """Tests for onboarding endpoints."""

    @pytest.mark.asyncio
    async def test_get_onboarding_questions(self, client: AsyncClient) -> None:
        """Test retrieval of onboarding questions."""
        response = await client.get("/api/v1/recommendations/onboarding/questions")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "questions" in data
        assert len(data["questions"]) > 0

        # Verify question structure
        question = data["questions"][0]
        assert "question_id" in question
        assert "question_text" in question
        assert "question_type" in question
        assert "required" in question

    @pytest.mark.asyncio
    async def test_submit_onboarding_success(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test successful onboarding submission."""
        onboarding_data = {
            "answers": [
                {"question_id": "risk_tolerance", "answer": "medium"},
                {"question_id": "investment_horizon", "answer": "long-term"},
                {
                    "question_id": "preferred_sectors",
                    "answer": ["Technology", "Healthcare"],
                },
            ]
        }

        response = await client.post(
            "/api/v1/recommendations/onboarding",
            json=onboarding_data,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["profile_created"] is True


class TestFeedLimitValidation:
    """Tests for feed limit validation."""

    @pytest.mark.asyncio
    async def test_feed_limit_too_low(self, client: AsyncClient, auth_token: str) -> None:
        """Test feed request with limit below minimum."""
        response = await client.get(
            "/api/v1/recommendations/feed?limit=0",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_feed_limit_too_high(self, client: AsyncClient, auth_token: str) -> None:
        """Test feed request with limit above maximum."""
        response = await client.get(
            "/api/v1/recommendations/feed?limit=150",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestFeedbackValidation:
    """Tests for feedback validation."""

    @pytest.mark.asyncio
    async def test_feedback_rating_validation(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test feedback with invalid rating value."""
        feedback_data = {
            "item_id": "signal_001",
            "feedback_type": "like",
            "rating": 6,
        }

        response = await client.post(
            "/api/v1/recommendations/feedback",
            json=feedback_data,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_feedback_comment_too_long(
        self, client: AsyncClient, auth_token: str
    ) -> None:
        """Test feedback with comment exceeding max length."""
        feedback_data = {
            "item_id": "signal_001",
            "feedback_type": "like",
            "comment": "x" * 501,
        }

        response = await client.post(
            "/api/v1/recommendations/feedback",
            json=feedback_data,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
