"""Comprehensive tests for API Gateway functionality."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.gateway.api_key_manager import APIKeyManager
from signalforge.gateway.tier_rate_limiter import TierRateLimiter
from signalforge.gateway.throttling import IntelligentThrottler, ThrottleLevel
from signalforge.gateway.usage_tracker import UsageTracker
from signalforge.models.api_key import APIKey, SubscriptionTier
from signalforge.models.user import User


@pytest.mark.asyncio
class TestAPIKeyManager:
    """Tests for API key management."""

    async def test_generate_api_key(self, db_session: AsyncSession, test_user: User) -> None:
        """Test API key generation."""
        plain_key, api_key = await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="Test Key",
            tier=SubscriptionTier.FREE,
            scopes=["read", "write"],
        )

        assert plain_key is not None
        assert len(plain_key) == 43
        assert api_key.user_id == test_user.id
        assert api_key.name == "Test Key"
        assert api_key.tier == SubscriptionTier.FREE
        assert api_key.scopes == ["read", "write"]
        assert api_key.is_active is True

    async def test_generate_api_key_with_expiration(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test API key generation with expiration."""
        expires_at = datetime.now(UTC) + timedelta(days=30)

        plain_key, api_key = await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="Expiring Key",
            tier=SubscriptionTier.PROSUMER,
            expires_at=expires_at,
        )

        assert api_key.expires_at is not None
        assert abs((api_key.expires_at - expires_at).total_seconds()) < 1

    async def test_generate_api_key_empty_name(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test API key generation with empty name."""
        with pytest.raises(ValueError, match="API key name cannot be empty"):
            await APIKeyManager.generate_api_key(
                db=db_session,
                user_id=test_user.id,
                name="   ",
                tier=SubscriptionTier.FREE,
            )

    async def test_validate_api_key_success(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test successful API key validation."""
        plain_key, created_key = await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="Valid Key",
            tier=SubscriptionTier.FREE,
        )

        validated_key = await APIKeyManager.validate_api_key(db_session, plain_key)

        assert validated_key is not None
        assert validated_key.id == created_key.id
        assert validated_key.user_id == test_user.id
        assert validated_key.last_used_at is not None

    async def test_validate_api_key_invalid(self, db_session: AsyncSession) -> None:
        """Test validation with invalid API key."""
        result = await APIKeyManager.validate_api_key(db_session, "invalid_key")
        assert result is None

    async def test_validate_api_key_expired(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test validation of expired API key."""
        expires_at = datetime.now(UTC) - timedelta(days=1)

        plain_key, _ = await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="Expired Key",
            tier=SubscriptionTier.FREE,
            expires_at=expires_at,
        )

        result = await APIKeyManager.validate_api_key(db_session, plain_key)
        assert result is None

    async def test_revoke_api_key(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test API key revocation."""
        _, api_key = await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="To Revoke",
            tier=SubscriptionTier.FREE,
        )

        success = await APIKeyManager.revoke_api_key(
            db_session,
            api_key.id,
            test_user.id,
        )

        assert success is True

        result = await db_session.execute(
            select(APIKey).where(APIKey.id == api_key.id)
        )
        revoked_key = result.scalar_one()
        assert revoked_key.is_active is False

    async def test_revoke_api_key_not_found(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test revocation of non-existent API key."""
        success = await APIKeyManager.revoke_api_key(
            db_session,
            uuid4(),
            test_user.id,
        )

        assert success is False

    async def test_list_user_keys(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test listing user API keys."""
        await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="Key 1",
            tier=SubscriptionTier.FREE,
        )

        await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="Key 2",
            tier=SubscriptionTier.PROSUMER,
        )

        keys = await APIKeyManager.list_user_keys(db_session, test_user.id)

        assert len(keys) == 2
        assert keys[0].name in ["Key 1", "Key 2"]
        assert keys[1].name in ["Key 1", "Key 2"]

    async def test_check_scope(
        self,
        db_session: AsyncSession,
        test_user: User,
    ) -> None:
        """Test scope checking."""
        _, api_key = await APIKeyManager.generate_api_key(
            db=db_session,
            user_id=test_user.id,
            name="Scoped Key",
            tier=SubscriptionTier.FREE,
            scopes=["read"],
        )

        has_read = await APIKeyManager.check_scope(api_key, "read")
        has_write = await APIKeyManager.check_scope(api_key, "write")

        assert has_read is True
        assert has_write is False


@pytest.mark.asyncio
class TestTierRateLimiter:
    """Tests for tier-based rate limiting."""

    async def test_rate_limit_free_tier(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test rate limiting for free tier."""
        rate_limiter = TierRateLimiter(redis_client)

        allowed, status = await rate_limiter.check_rate_limit(
            test_user.id,
            SubscriptionTier.FREE,
        )

        assert allowed is True
        assert status.tier == SubscriptionTier.FREE.value
        assert status.limit == 5

    async def test_rate_limit_burst_exceeded(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test burst limit exceeded."""
        rate_limiter = TierRateLimiter(redis_client)

        # Use burst_limit_override=3 to ensure burst is hit before sustained (5)
        for _ in range(3):
            await rate_limiter.check_rate_limit(
                test_user.id, SubscriptionTier.FREE, burst_limit_override=3
            )

        allowed, status = await rate_limiter.check_rate_limit(
            test_user.id,
            SubscriptionTier.FREE,
            burst_limit_override=3,
        )

        assert allowed is False
        assert status.is_burst is True

    async def test_rate_limit_sustained_exceeded(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test sustained limit exceeded."""
        rate_limiter = TierRateLimiter(redis_client)

        for _ in range(6):
            await rate_limiter.check_rate_limit(test_user.id, SubscriptionTier.FREE)

        allowed, status = await rate_limiter.check_rate_limit(
            test_user.id,
            SubscriptionTier.FREE,
        )

        assert allowed is False
        assert status.is_burst is False

    async def test_rate_limit_prosumer_tier(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test rate limiting for prosumer tier."""
        rate_limiter = TierRateLimiter(redis_client)

        for _ in range(50):
            allowed, _ = await rate_limiter.check_rate_limit(
                test_user.id,
                SubscriptionTier.PROSUMER,
            )
            assert allowed is True

    async def test_rate_limit_override(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test rate limit override."""
        rate_limiter = TierRateLimiter(redis_client)

        allowed, status = await rate_limiter.check_rate_limit(
            test_user.id,
            SubscriptionTier.FREE,
            rate_limit_override=100,
        )

        assert allowed is True
        assert status.limit == 100

    async def test_get_rate_limit_status(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test getting rate limit status."""
        rate_limiter = TierRateLimiter(redis_client)

        await rate_limiter.check_rate_limit(test_user.id, SubscriptionTier.FREE)
        await rate_limiter.check_rate_limit(test_user.id, SubscriptionTier.FREE)

        status = await rate_limiter.get_rate_limit_status(
            test_user.id,
            SubscriptionTier.FREE,
        )

        assert status.used == 2
        assert status.remaining == 3

    async def test_reset_rate_limit(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test resetting rate limit."""
        rate_limiter = TierRateLimiter(redis_client)

        for _ in range(5):
            await rate_limiter.check_rate_limit(test_user.id, SubscriptionTier.FREE)

        await rate_limiter.reset_rate_limit(test_user.id)

        allowed, status = await rate_limiter.check_rate_limit(
            test_user.id,
            SubscriptionTier.FREE,
        )

        assert allowed is True
        assert status.used == 1


@pytest.mark.asyncio
class TestUsageTracker:
    """Tests for usage tracking."""

    async def test_track_request(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test request tracking."""
        tracker = UsageTracker(redis_client)

        await tracker.track_request(
            user_id=test_user.id,
            endpoint="/api/v1/test",
            response_time_ms=150.5,
            status_code=200,
            data_transferred_bytes=1024,
        )

        stats = await tracker.get_usage_stats(test_user.id, "today")

        assert stats.total_requests == 1
        assert stats.successful_requests == 1

    async def test_track_multiple_requests(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test tracking multiple requests."""
        tracker = UsageTracker(redis_client)

        for i in range(5):
            await tracker.track_request(
                user_id=test_user.id,
                endpoint=f"/api/v1/endpoint{i % 2}",
                response_time_ms=100.0,
                status_code=200,
            )

        stats = await tracker.get_usage_stats(test_user.id, "today")

        assert stats.total_requests == 5
        assert len(stats.endpoints_used) > 0

    async def test_track_failed_request(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test tracking failed requests."""
        tracker = UsageTracker(redis_client)

        await tracker.track_request(
            user_id=test_user.id,
            endpoint="/api/v1/test",
            response_time_ms=50.0,
            status_code=500,
        )

        stats = await tracker.get_usage_stats(test_user.id, "today")

        assert stats.failed_requests == 1

    async def test_track_rate_limited_request(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test tracking rate-limited requests."""
        tracker = UsageTracker(redis_client)

        await tracker.track_request(
            user_id=test_user.id,
            endpoint="/api/v1/test",
            response_time_ms=10.0,
            status_code=429,
        )

        stats = await tracker.get_usage_stats(test_user.id, "today")

        assert stats.rate_limited_requests == 1

    async def test_get_endpoint_breakdown(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test endpoint usage breakdown."""
        tracker = UsageTracker(redis_client)

        for _ in range(3):
            await tracker.track_request(
                test_user.id,
                "/api/v1/endpoint1",
                100.0,
                200,
            )

        for _ in range(2):
            await tracker.track_request(
                test_user.id,
                "/api/v1/endpoint2",
                100.0,
                200,
            )

        breakdown = await tracker.get_endpoint_breakdown(test_user.id, days=1)

        assert breakdown["/api/v1/endpoint1"] == 3
        assert breakdown["/api/v1/endpoint2"] == 2


@pytest.mark.asyncio
class TestIntelligentThrottler:
    """Tests for intelligent throttling."""

    async def test_normal_throttle_level(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test normal throttle level."""
        throttler = IntelligentThrottler(redis_client)

        status = await throttler.check_throttle(
            test_user.id,
            SubscriptionTier.FREE,
            "/api/v1/test",
        )

        assert status.level == ThrottleLevel.NORMAL
        assert status.should_process is True
        assert status.delay_ms == 0

    async def test_increment_decrement_load(self, redis_client) -> None:
        """Test load increment and decrement."""
        throttler = IntelligentThrottler(redis_client)

        await throttler.increment_load(10.0)
        load = await throttler.get_system_load()
        assert load == 10.0

        await throttler.decrement_load(5.0)
        load = await throttler.get_system_load()
        assert load == 5.0

    async def test_professional_tier_priority(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test that professional tier has higher priority."""
        throttler = IntelligentThrottler(redis_client)

        await throttler.increment_load(80.0)

        pro_status = await throttler.check_throttle(
            test_user.id,
            SubscriptionTier.PROFESSIONAL,
            "/api/v1/test",
        )

        free_status = await throttler.check_throttle(
            test_user.id,
            SubscriptionTier.FREE,
            "/api/v1/test",
        )

        assert pro_status.delay_ms < free_status.delay_ms

    async def test_enqueue_dequeue_request(
        self,
        redis_client,
        test_user: User,
    ) -> None:
        """Test request queuing."""
        throttler = IntelligentThrottler(redis_client)

        await throttler.enqueue_request(
            test_user.id,
            SubscriptionTier.FREE,
            "request-123",
        )

        result = await throttler.dequeue_request(SubscriptionTier.FREE)

        assert result is not None
        user_id, request_id = result
        assert user_id == test_user.id
        assert request_id == "request-123"


@pytest.mark.asyncio
class TestAPIKeyRoutes:
    """Tests for API key routes."""

    async def test_create_api_key_endpoint(
        self,
        client: AsyncClient,
        authenticated_user_token: str,
    ) -> None:
        """Test API key creation endpoint."""
        response = await client.post(
            "/api/v1/api-keys",
            json={
                "name": "Test API Key",
                "tier": "free",
                "scopes": ["read"],
            },
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "api_key" in data["data"]
        assert data["data"]["key_info"]["name"] == "Test API Key"

    async def test_list_api_keys_endpoint(
        self,
        client: AsyncClient,
        authenticated_user_token: str,
    ) -> None:
        """Test listing API keys endpoint."""
        await client.post(
            "/api/v1/api-keys",
            json={"name": "Key 1", "tier": "free"},
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        response = await client.get(
            "/api/v1/api-keys",
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1

    async def test_revoke_api_key_endpoint(
        self,
        client: AsyncClient,
        authenticated_user_token: str,
    ) -> None:
        """Test API key revocation endpoint."""
        create_response = await client.post(
            "/api/v1/api-keys",
            json={"name": "To Revoke", "tier": "free"},
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        key_id = create_response.json()["data"]["key_info"]["id"]

        response = await client.delete(
            f"/api/v1/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    async def test_get_api_key_usage_endpoint(
        self,
        client: AsyncClient,
        authenticated_user_token: str,
    ) -> None:
        """Test getting API key usage endpoint."""
        create_response = await client.post(
            "/api/v1/api-keys",
            json={"name": "Usage Key", "tier": "free"},
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        key_id = create_response.json()["data"]["key_info"]["id"]

        response = await client.get(
            f"/api/v1/api-keys/{key_id}/usage?period=today",
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_requests" in data["data"]

    async def test_get_rate_limit_status_endpoint(
        self,
        client: AsyncClient,
        authenticated_user_token: str,
    ) -> None:
        """Test getting rate limit status endpoint."""
        create_response = await client.post(
            "/api/v1/api-keys",
            json={"name": "Rate Limit Key", "tier": "free"},
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        key_id = create_response.json()["data"]["key_info"]["id"]

        response = await client.get(
            f"/api/v1/api-keys/{key_id}/rate-limit-status",
            headers={"Authorization": f"Bearer {authenticated_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "limit" in data["data"]
        assert "remaining" in data["data"]
