"""Comprehensive tests for billing and subscription system."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.billing.models import (
    BillingCycle,
    Invoice,
    InvoiceStatus,
    Subscription,
    SubscriptionPlan,
    SubscriptionStatus,
    UsageRecord,
)
from signalforge.billing.quota_manager import QuotaExceededError, QuotaManager
from signalforge.billing.schemas import (
    SubscriptionCancelRequest,
    SubscriptionCreate,
    SubscriptionPlanCreate,
)
from signalforge.billing.service import (
    BillingService,
    InvalidSubscriptionError,
    PlanNotFoundError,
)
from signalforge.billing.tier_features import (
    TIER_FEATURES,
    SubscriptionTier,
    get_quota_limit,
    get_tier_feature,
    has_unlimited_quota,
    is_feature_enabled,
)
from signalforge.core.security import create_access_token, get_password_hash
from signalforge.models.user import User


@pytest.fixture
def mock_redis() -> MagicMock:
    """Create mock Redis client."""
    mock = MagicMock(spec=Redis)
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.scan = AsyncMock(return_value=(0, []))
    return mock


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpass123"),
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def free_plan(db_session: AsyncSession) -> SubscriptionPlan:
    """Create free tier subscription plan."""
    plan = SubscriptionPlan(
        name="Free Plan",
        tier=SubscriptionTier.FREE,
        description="Basic free tier",
        price_monthly=Decimal("0.00"),
        price_yearly=Decimal("0.00"),
        features=TIER_FEATURES["free"],
        is_active=True,
        trial_days=0,
    )
    db_session.add(plan)
    await db_session.flush()
    await db_session.refresh(plan)
    return plan


@pytest.fixture
async def prosumer_plan(db_session: AsyncSession) -> SubscriptionPlan:
    """Create prosumer tier subscription plan."""
    plan = SubscriptionPlan(
        name="Prosumer Plan",
        tier=SubscriptionTier.PROSUMER,
        description="For active investors",
        price_monthly=Decimal("29.99"),
        price_yearly=Decimal("299.99"),
        features=TIER_FEATURES["prosumer"],
        is_active=True,
        trial_days=14,
    )
    db_session.add(plan)
    await db_session.flush()
    await db_session.refresh(plan)
    return plan


@pytest.fixture
async def professional_plan(db_session: AsyncSession) -> SubscriptionPlan:
    """Create professional tier subscription plan."""
    plan = SubscriptionPlan(
        name="Professional Plan",
        tier=SubscriptionTier.PROFESSIONAL,
        description="For professional traders",
        price_monthly=Decimal("99.99"),
        price_yearly=Decimal("999.99"),
        features=TIER_FEATURES["professional"],
        is_active=True,
        trial_days=7,
    )
    db_session.add(plan)
    await db_session.flush()
    await db_session.refresh(plan)
    return plan


class TestTierFeatures:
    """Test tier feature configuration."""

    def test_tier_features_structure(self) -> None:
        """Test TIER_FEATURES has correct structure."""
        assert "free" in TIER_FEATURES
        assert "prosumer" in TIER_FEATURES
        assert "professional" in TIER_FEATURES

        for tier_name, features in TIER_FEATURES.items():
            assert "predictions_per_day" in features
            assert "api_calls_per_minute" in features
            assert "history_days" in features
            assert "nlp_summaries" in features
            assert "sector_reports" in features
            assert "bulk_api" in features

    def test_get_tier_feature(self) -> None:
        """Test getting specific tier features."""
        predictions = get_tier_feature(SubscriptionTier.FREE, "predictions_per_day")
        assert predictions == 10

        predictions = get_tier_feature(SubscriptionTier.PROSUMER, "predictions_per_day")
        assert predictions == 100

        predictions = get_tier_feature(
            SubscriptionTier.PROFESSIONAL, "predictions_per_day"
        )
        assert predictions == -1  # unlimited

    def test_get_tier_feature_invalid(self) -> None:
        """Test getting invalid tier feature raises error."""
        with pytest.raises(KeyError):
            get_tier_feature(SubscriptionTier.FREE, "nonexistent_feature")

    def test_is_feature_enabled(self) -> None:
        """Test feature enabled check."""
        assert not is_feature_enabled(SubscriptionTier.FREE, "nlp_summaries")
        assert is_feature_enabled(SubscriptionTier.PROSUMER, "nlp_summaries")
        assert is_feature_enabled(SubscriptionTier.PROFESSIONAL, "white_label")

    def test_get_quota_limit(self) -> None:
        """Test quota limit retrieval."""
        assert get_quota_limit(SubscriptionTier.FREE, "predictions_per_day") == 10
        assert get_quota_limit(SubscriptionTier.PROSUMER, "api_calls_per_minute") == 60
        assert (
            get_quota_limit(SubscriptionTier.PROFESSIONAL, "predictions_per_day") == -1
        )

    def test_has_unlimited_quota(self) -> None:
        """Test unlimited quota check."""
        assert not has_unlimited_quota(SubscriptionTier.FREE, "predictions_per_day")
        assert not has_unlimited_quota(SubscriptionTier.PROSUMER, "predictions_per_day")
        assert has_unlimited_quota(
            SubscriptionTier.PROFESSIONAL, "predictions_per_day"
        )
        assert has_unlimited_quota(SubscriptionTier.PROFESSIONAL, "history_days")


class TestSubscriptionModels:
    """Test subscription models."""

    async def test_subscription_plan_creation(
        self, db_session: AsyncSession, free_plan: SubscriptionPlan
    ) -> None:
        """Test subscription plan creation."""
        assert free_plan.id is not None
        assert free_plan.tier == SubscriptionTier.FREE
        assert free_plan.price_monthly == Decimal("0.00")
        assert free_plan.is_active is True

    async def test_subscription_creation(
        self,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
    ) -> None:
        """Test subscription creation."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
        )
        db_session.add(subscription)
        await db_session.flush()
        await db_session.refresh(subscription)

        assert subscription.id is not None
        assert subscription.user_id == test_user.id
        assert subscription.plan_id == prosumer_plan.id
        assert subscription.is_active is True

    async def test_subscription_is_active_property(
        self, db_session: AsyncSession, test_user: User, free_plan: SubscriptionPlan
    ) -> None:
        """Test subscription is_active property."""
        now = datetime.now(UTC)

        active_sub = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=now,
            expires_at=now + timedelta(days=30),
        )
        assert active_sub.is_active is True

        expired_sub = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=now - timedelta(days=60),
            expires_at=now - timedelta(days=30),
        )
        assert expired_sub.is_active is False

        canceled_sub = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.CANCELED,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=now,
            expires_at=now + timedelta(days=30),
        )
        assert canceled_sub.is_active is False

    async def test_subscription_days_until_expiration(
        self, test_user: User, free_plan: SubscriptionPlan
    ) -> None:
        """Test days until expiration calculation."""
        now = datetime.now(UTC)

        sub_with_expiration = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=now,
            expires_at=now + timedelta(days=15),
        )
        # Property uses datetime.now() which may be slightly later than test's now
        # This can cause a 1 day difference due to timing
        assert sub_with_expiration.days_until_expiration in (14, 15)

        sub_no_expiration = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=now,
            expires_at=None,
        )
        assert sub_no_expiration.days_until_expiration is None

    async def test_invoice_creation(
        self, db_session: AsyncSession, test_user: User, prosumer_plan: SubscriptionPlan
    ) -> None:
        """Test invoice creation."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        now = datetime.now(UTC)
        invoice = Invoice(
            subscription_id=subscription.id,
            invoice_number="INV-001",
            amount=Decimal("29.99"),
            status=InvoiceStatus.PENDING,
            billing_period_start=now,
            billing_period_end=now + timedelta(days=30),
            due_date=now + timedelta(days=7),
        )
        db_session.add(invoice)
        await db_session.flush()
        await db_session.refresh(invoice)

        assert invoice.id is not None
        assert invoice.invoice_number == "INV-001"
        assert invoice.amount == Decimal("29.99")
        assert invoice.is_paid is False
        assert invoice.is_overdue is False

    async def test_usage_record_creation(
        self, db_session: AsyncSession, test_user: User
    ) -> None:
        """Test usage record creation."""
        usage = UsageRecord(
            user_id=test_user.id,
            resource_type="predictions_per_day",
            count=5,
        )
        db_session.add(usage)
        await db_session.flush()
        await db_session.refresh(usage)

        assert usage.id is not None
        assert usage.user_id == test_user.id
        assert usage.resource_type == "predictions_per_day"
        assert usage.count == 5


class TestBillingService:
    """Test billing service."""

    async def test_create_plan(self, db_session: AsyncSession) -> None:
        """Test plan creation."""
        service = BillingService(db_session)
        plan_data = SubscriptionPlanCreate(
            name="Test Plan",
            tier=SubscriptionTier.FREE,
            description="Test description",
            price_monthly=Decimal("0.00"),
            price_yearly=Decimal("0.00"),
            features=TIER_FEATURES["free"],
            is_active=True,
            trial_days=0,
        )

        plan = await service.create_plan(plan_data)

        assert plan.id is not None
        assert plan.name == "Test Plan"
        assert plan.tier == SubscriptionTier.FREE

    async def test_get_plan(
        self, db_session: AsyncSession, free_plan: SubscriptionPlan
    ) -> None:
        """Test getting a plan."""
        service = BillingService(db_session)
        plan = await service.get_plan(free_plan.id)

        assert plan.id == free_plan.id
        assert plan.tier == SubscriptionTier.FREE

    async def test_get_plan_not_found(self, db_session: AsyncSession) -> None:
        """Test getting non-existent plan raises error."""
        service = BillingService(db_session)

        with pytest.raises(PlanNotFoundError):
            await service.get_plan(uuid4())

    async def test_get_plan_by_tier(
        self, db_session: AsyncSession, prosumer_plan: SubscriptionPlan
    ) -> None:
        """Test getting plan by tier."""
        service = BillingService(db_session)
        plan = await service.get_plan_by_tier(SubscriptionTier.PROSUMER)

        assert plan.id == prosumer_plan.id
        assert plan.tier == SubscriptionTier.PROSUMER

    async def test_list_plans(
        self,
        db_session: AsyncSession,
        free_plan: SubscriptionPlan,
        prosumer_plan: SubscriptionPlan,
        professional_plan: SubscriptionPlan,
    ) -> None:
        """Test listing all plans."""
        service = BillingService(db_session)
        plans = await service.list_plans()

        assert len(plans) == 3
        assert all(plan.is_active for plan in plans)

    async def test_subscribe_user(
        self, db_session: AsyncSession, test_user: User, prosumer_plan: SubscriptionPlan
    ) -> None:
        """Test subscribing a user to a plan."""
        service = BillingService(db_session)
        subscription_data = SubscriptionCreate(
            plan_id=prosumer_plan.id,
            billing_cycle=BillingCycle.MONTHLY,
            auto_renew=True,
        )

        subscription = await service.subscribe_user(test_user.id, subscription_data)

        assert subscription.id is not None
        assert subscription.user_id == test_user.id
        assert subscription.plan_id == prosumer_plan.id
        assert subscription.status == SubscriptionStatus.TRIALING
        assert subscription.trial_ends_at is not None

    async def test_subscribe_user_with_existing_subscription(
        self, db_session: AsyncSession, test_user: User, prosumer_plan: SubscriptionPlan
    ) -> None:
        """Test subscribing user who already has active subscription.

        Note: The current implementation only checks for ACTIVE subscriptions,
        not TRIALING. So we need to manually set the status to ACTIVE to test
        the duplicate subscription prevention logic.
        """
        service = BillingService(db_session)

        subscription_data = SubscriptionCreate(
            plan_id=prosumer_plan.id,
            billing_cycle=BillingCycle.MONTHLY,
            auto_renew=True,
        )
        sub = await service.subscribe_user(test_user.id, subscription_data)

        # Manually set to ACTIVE to test the duplicate check
        sub.status = SubscriptionStatus.ACTIVE
        await db_session.commit()

        with pytest.raises(InvalidSubscriptionError):
            await service.subscribe_user(test_user.id, subscription_data)

    async def test_cancel_subscription(
        self, db_session: AsyncSession, test_user: User, prosumer_plan: SubscriptionPlan
    ) -> None:
        """Test canceling a subscription."""
        service = BillingService(db_session)

        subscription_data = SubscriptionCreate(
            plan_id=prosumer_plan.id,
            billing_cycle=BillingCycle.MONTHLY,
            auto_renew=True,
        )
        subscription = await service.subscribe_user(test_user.id, subscription_data)

        canceled = await service.cancel_subscription(
            test_user.id,
            subscription.id,
            cancel_immediately=False,
            reason="Testing cancellation",
        )

        assert canceled.status == SubscriptionStatus.CANCELED
        assert canceled.canceled_at is not None
        assert canceled.auto_renew is False
        assert canceled.extra_data is not None
        assert canceled.extra_data.get("cancellation_reason") == "Testing cancellation"

    async def test_reactivate_subscription(
        self, db_session: AsyncSession, test_user: User, prosumer_plan: SubscriptionPlan
    ) -> None:
        """Test reactivating a canceled subscription."""
        service = BillingService(db_session)

        subscription_data = SubscriptionCreate(
            plan_id=prosumer_plan.id,
            billing_cycle=BillingCycle.MONTHLY,
            auto_renew=True,
        )
        subscription = await service.subscribe_user(test_user.id, subscription_data)
        await service.cancel_subscription(
            test_user.id, subscription.id, cancel_immediately=False
        )

        reactivated = await service.reactivate_subscription(
            test_user.id, subscription.id
        )

        assert reactivated.status == SubscriptionStatus.ACTIVE
        assert reactivated.canceled_at is None
        assert reactivated.auto_renew is True


class TestQuotaManager:
    """Test quota manager."""

    async def test_get_user_tier_no_subscription(
        self, db_session: AsyncSession, test_user: User, mock_redis: MagicMock
    ) -> None:
        """Test getting tier for user without subscription."""
        manager = QuotaManager(db_session, mock_redis)
        tier = await manager.get_user_tier(test_user.id)

        assert tier == SubscriptionTier.FREE

    async def test_get_user_tier_with_subscription(
        self,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test getting tier for user with subscription."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)
        tier = await manager.get_user_tier(test_user.id)

        assert tier == SubscriptionTier.PROSUMER

    async def test_check_quota_unlimited(
        self,
        db_session: AsyncSession,
        test_user: User,
        professional_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test quota check for unlimited resource."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=professional_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)
        allowed = await manager.check_quota(test_user.id, "predictions_per_day", 1000)

        assert allowed is True

    async def test_check_quota_within_limit(
        self,
        db_session: AsyncSession,
        test_user: User,
        free_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test quota check within limit."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)
        allowed = await manager.check_quota(test_user.id, "predictions_per_day", 5)

        assert allowed is True

    async def test_check_quota_exceeds_limit(
        self,
        db_session: AsyncSession,
        test_user: User,
        free_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test quota check exceeding limit."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)

        for _ in range(10):
            await manager.increment_usage(test_user.id, "predictions_per_day", 1)

        allowed = await manager.check_quota(test_user.id, "predictions_per_day", 1)
        assert allowed is False

    async def test_increment_usage(
        self,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test incrementing usage."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)
        usage_record = await manager.increment_usage(
            test_user.id, "predictions_per_day", 5
        )

        assert usage_record.id is not None
        assert usage_record.user_id == test_user.id
        assert usage_record.count == 5

    async def test_increment_usage_exceeds_quota(
        self,
        db_session: AsyncSession,
        test_user: User,
        free_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test incrementing usage that exceeds quota."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)

        with pytest.raises(QuotaExceededError):
            await manager.increment_usage(test_user.id, "predictions_per_day", 15)

    async def test_get_current_usage(
        self,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test getting current usage."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)
        await manager.increment_usage(test_user.id, "predictions_per_day", 3)
        await manager.increment_usage(test_user.id, "predictions_per_day", 2)

        usage = await manager.get_current_usage(test_user.id, "predictions_per_day")
        assert usage == 5

    async def test_get_remaining_quota(
        self,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test getting remaining quota."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)
        await manager.increment_usage(test_user.id, "predictions_per_day", 25)

        remaining = await manager.get_remaining_quota(
            test_user.id, "predictions_per_day"
        )
        assert remaining == 75

    async def test_get_usage_stats(
        self,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
        mock_redis: MagicMock,
    ) -> None:
        """Test getting usage statistics."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        manager = QuotaManager(db_session, mock_redis)
        await manager.increment_usage(test_user.id, "predictions_per_day", 10)

        stats = await manager.get_usage_stats(test_user.id)

        assert "predictions_per_day" in stats
        assert stats["predictions_per_day"]["current_usage"] == 10
        assert stats["predictions_per_day"]["quota_limit"] == 100
        assert stats["predictions_per_day"]["remaining"] == 90
        assert stats["predictions_per_day"]["is_unlimited"] is False


class TestBillingAPI:
    """Test billing API endpoints."""

    async def test_list_plans(
        self,
        client: AsyncClient,
        free_plan: SubscriptionPlan,
        prosumer_plan: SubscriptionPlan,
    ) -> None:
        """Test listing subscription plans."""
        response = await client.get("/api/v1/billing/plans")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    async def test_get_current_subscription_no_subscription(
        self, client: AsyncClient, test_user: User
    ) -> None:
        """Test getting current subscription when user has none."""
        token = create_access_token(data={"sub": str(test_user.id)})
        response = await client.get(
            "/api/v1/billing/subscription",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json() is None

    async def test_subscribe_to_plan(
        self, client: AsyncClient, test_user: User, prosumer_plan: SubscriptionPlan
    ) -> None:
        """Test subscribing to a plan."""
        token = create_access_token(data={"sub": str(test_user.id)})
        response = await client.post(
            "/api/v1/billing/subscribe",
            json={
                "plan_id": str(prosumer_plan.id),
                "billing_cycle": "monthly",
                "auto_renew": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["plan_id"] == str(prosumer_plan.id)
        assert data["status"] == "trialing"

    async def test_cancel_subscription(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
    ) -> None:
        """Test canceling a subscription."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        token = create_access_token(data={"sub": str(test_user.id)})
        response = await client.post(
            "/api/v1/billing/cancel",
            json={"cancel_immediately": False, "reason": "Test cancellation"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "canceled"

    async def test_get_usage_stats(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        test_user: User,
        prosumer_plan: SubscriptionPlan,
    ) -> None:
        """Test getting usage statistics."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=prosumer_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        token = create_access_token(data={"sub": str(test_user.id)})
        response = await client.get(
            "/api/v1/billing/usage",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tier"] == "prosumer"
        assert "usage" in data
        assert len(data["usage"]) > 0

    async def test_check_quota(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        test_user: User,
        free_plan: SubscriptionPlan,
    ) -> None:
        """Test quota check endpoint."""
        subscription = Subscription(
            user_id=test_user.id,
            plan_id=free_plan.id,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=BillingCycle.MONTHLY,
            started_at=datetime.now(UTC),
        )
        db_session.add(subscription)
        await db_session.flush()

        token = create_access_token(data={"sub": str(test_user.id)})
        response = await client.post(
            "/api/v1/billing/quota/check",
            json={"resource_type": "predictions_per_day"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["allowed"] is True
        assert data["resource_type"] == "predictions_per_day"
        assert data["quota_limit"] == 10
