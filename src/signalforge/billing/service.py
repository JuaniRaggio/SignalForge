"""Billing service for subscription management."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from signalforge.billing.models import (
    BillingCycle,
    Invoice,
    InvoiceStatus,
    Subscription,
    SubscriptionPlan,
    SubscriptionStatus,
)
from signalforge.billing.schemas import (
    InvoiceCreate,
    SubscriptionCreate,
    SubscriptionPlanCreate,
)
from signalforge.billing.tier_features import SubscriptionTier
from signalforge.core.logging import LoggerMixin


class SubscriptionNotFoundError(Exception):
    """Exception raised when subscription is not found."""



class PlanNotFoundError(Exception):
    """Exception raised when plan is not found."""



class InvalidSubscriptionError(Exception):
    """Exception raised for invalid subscription operations."""



class BillingService(LoggerMixin):
    """Service for managing subscriptions and billing."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize billing service.

        Args:
            db: Database session
        """
        self.db = db

    async def create_plan(
        self,
        plan_data: SubscriptionPlanCreate,
    ) -> SubscriptionPlan:
        """Create a new subscription plan.

        Args:
            plan_data: Plan creation data

        Returns:
            Created subscription plan

        Raises:
            IntegrityError: If plan with tier already exists
        """
        plan = SubscriptionPlan(
            name=plan_data.name,
            tier=plan_data.tier,
            description=plan_data.description,
            price_monthly=plan_data.price_monthly,
            price_yearly=plan_data.price_yearly,
            currency=plan_data.currency,
            features=plan_data.features,
            is_active=plan_data.is_active,
            trial_days=plan_data.trial_days,
        )

        self.db.add(plan)
        await self.db.flush()
        await self.db.refresh(plan)

        self.logger.info(
            "subscription_plan_created",
            plan_id=str(plan.id),
            tier=plan.tier.value,
        )

        return plan

    async def get_plan(self, plan_id: UUID) -> SubscriptionPlan:
        """Get a subscription plan by ID.

        Args:
            plan_id: Plan ID

        Returns:
            Subscription plan

        Raises:
            PlanNotFoundError: If plan not found
        """
        result = await self.db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.id == plan_id),
        )
        plan = result.scalar_one_or_none()

        if plan is None:
            raise PlanNotFoundError(f"Plan {plan_id} not found")

        return plan

    async def get_plan_by_tier(self, tier: SubscriptionTier) -> SubscriptionPlan:
        """Get a subscription plan by tier.

        Args:
            tier: Subscription tier

        Returns:
            Subscription plan

        Raises:
            PlanNotFoundError: If plan not found
        """
        result = await self.db.execute(
            select(SubscriptionPlan).where(
                SubscriptionPlan.tier == tier,
                SubscriptionPlan.is_active.is_(True),
            ),
        )
        plan = result.scalar_one_or_none()

        if plan is None:
            raise PlanNotFoundError(f"No active plan found for tier {tier.value}")

        return plan

    async def list_plans(self, active_only: bool = True) -> list[SubscriptionPlan]:
        """List all subscription plans.

        Args:
            active_only: Only return active plans

        Returns:
            List of subscription plans
        """
        query = select(SubscriptionPlan)
        if active_only:
            query = query.where(SubscriptionPlan.is_active.is_(True))

        result = await self.db.execute(query.order_by(SubscriptionPlan.price_monthly))
        return list(result.scalars().all())

    async def subscribe_user(
        self,
        user_id: UUID,
        subscription_data: SubscriptionCreate,
    ) -> Subscription:
        """Subscribe a user to a plan.

        Args:
            user_id: User ID
            subscription_data: Subscription data

        Returns:
            Created subscription

        Raises:
            PlanNotFoundError: If plan not found
            InvalidSubscriptionError: If user already has active subscription
        """
        plan = await self.get_plan(subscription_data.plan_id)

        existing = await self.get_user_active_subscription(user_id)
        if existing:
            raise InvalidSubscriptionError(
                "User already has an active subscription. Cancel it first.",
            )

        now = datetime.now(UTC)
        trial_ends_at = None
        if plan.trial_days > 0:
            trial_ends_at = now + timedelta(days=plan.trial_days)
            status = SubscriptionStatus.TRIALING
        else:
            status = SubscriptionStatus.ACTIVE

        expires_at = self._calculate_expiration_date(
            now,
            subscription_data.billing_cycle,
        )

        subscription = Subscription(
            user_id=user_id,
            plan_id=plan.id,
            status=status,
            billing_cycle=subscription_data.billing_cycle,
            started_at=now,
            expires_at=expires_at,
            trial_ends_at=trial_ends_at,
            auto_renew=subscription_data.auto_renew,
        )

        self.db.add(subscription)
        await self.db.flush()
        await self.db.refresh(subscription)

        self.logger.info(
            "user_subscribed",
            user_id=str(user_id),
            subscription_id=str(subscription.id),
            plan_id=str(plan.id),
            tier=plan.tier.value,
            status=status.value,
        )

        return subscription

    async def get_user_subscription(
        self,
        user_id: UUID,
        subscription_id: UUID,
    ) -> Subscription:
        """Get a specific subscription for a user.

        Args:
            user_id: User ID
            subscription_id: Subscription ID

        Returns:
            Subscription

        Raises:
            SubscriptionNotFoundError: If subscription not found
        """
        result = await self.db.execute(
            select(Subscription)
            .where(
                Subscription.id == subscription_id,
                Subscription.user_id == user_id,
            )
            .options(),
        )
        subscription = result.scalar_one_or_none()

        if subscription is None:
            raise SubscriptionNotFoundError(
                f"Subscription {subscription_id} not found for user {user_id}",
            )

        return subscription

    async def get_user_active_subscription(
        self,
        user_id: UUID,
    ) -> Subscription | None:
        """Get user's current active subscription.

        Args:
            user_id: User ID

        Returns:
            Active subscription or None
        """
        result = await self.db.execute(
            select(Subscription)
            .where(
                Subscription.user_id == user_id,
                Subscription.status == SubscriptionStatus.ACTIVE,
            )
            .order_by(Subscription.created_at.desc()),
        )
        return result.scalar_one_or_none()

    async def cancel_subscription(
        self,
        user_id: UUID,
        subscription_id: UUID,
        cancel_immediately: bool = False,
        reason: str | None = None,
    ) -> Subscription:
        """Cancel a subscription.

        Args:
            user_id: User ID
            subscription_id: Subscription ID
            cancel_immediately: If True, cancel immediately; otherwise at period end
            reason: Cancellation reason

        Returns:
            Updated subscription

        Raises:
            SubscriptionNotFoundError: If subscription not found
            InvalidSubscriptionError: If subscription cannot be canceled
        """
        subscription = await self.get_user_subscription(user_id, subscription_id)

        if subscription.status not in [
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIALING,
        ]:
            raise InvalidSubscriptionError(
                f"Cannot cancel subscription with status {subscription.status.value}",
            )

        now = datetime.now(UTC)
        subscription.canceled_at = now
        subscription.auto_renew = False

        if cancel_immediately:
            subscription.status = SubscriptionStatus.CANCELED
            subscription.expires_at = now
        else:
            subscription.status = SubscriptionStatus.CANCELED

        if reason:
            if subscription.extra_data is None:
                subscription.extra_data = {}
            subscription.extra_data["cancellation_reason"] = reason
            # Mark extra_data as modified for SQLAlchemy to track the change
            flag_modified(subscription, "extra_data")

        await self.db.flush()
        await self.db.refresh(subscription)

        self.logger.info(
            "subscription_canceled",
            user_id=str(user_id),
            subscription_id=str(subscription_id),
            cancel_immediately=cancel_immediately,
        )

        return subscription

    async def reactivate_subscription(
        self,
        user_id: UUID,
        subscription_id: UUID,
    ) -> Subscription:
        """Reactivate a canceled subscription.

        Args:
            user_id: User ID
            subscription_id: Subscription ID

        Returns:
            Updated subscription

        Raises:
            SubscriptionNotFoundError: If subscription not found
            InvalidSubscriptionError: If subscription cannot be reactivated
        """
        subscription = await self.get_user_subscription(user_id, subscription_id)

        if subscription.status != SubscriptionStatus.CANCELED:
            raise InvalidSubscriptionError(
                f"Cannot reactivate subscription with status {subscription.status.value}",
            )

        if subscription.expires_at and subscription.expires_at < datetime.now(UTC):
            raise InvalidSubscriptionError("Cannot reactivate expired subscription")

        subscription.status = SubscriptionStatus.ACTIVE
        subscription.canceled_at = None
        subscription.auto_renew = True

        await self.db.flush()
        await self.db.refresh(subscription)

        self.logger.info(
            "subscription_reactivated",
            user_id=str(user_id),
            subscription_id=str(subscription_id),
        )

        return subscription

    async def create_invoice(
        self,
        invoice_data: InvoiceCreate,
    ) -> Invoice:
        """Create an invoice for a subscription.

        Args:
            invoice_data: Invoice creation data

        Returns:
            Created invoice
        """
        invoice = Invoice(
            subscription_id=invoice_data.subscription_id,
            invoice_number=invoice_data.invoice_number,
            amount=invoice_data.amount,
            currency=invoice_data.currency,
            status=InvoiceStatus.PENDING,
            billing_period_start=invoice_data.billing_period_start,
            billing_period_end=invoice_data.billing_period_end,
            due_date=invoice_data.due_date,
            notes=invoice_data.notes,
        )

        self.db.add(invoice)
        await self.db.flush()
        await self.db.refresh(invoice)

        self.logger.info(
            "invoice_created",
            invoice_id=str(invoice.id),
            subscription_id=str(invoice.subscription_id),
            amount=str(invoice.amount),
        )

        return invoice

    async def get_user_invoices(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Invoice]:
        """Get invoices for a user.

        Args:
            user_id: User ID
            limit: Maximum number of invoices to return
            offset: Number of invoices to skip

        Returns:
            List of invoices
        """
        result = await self.db.execute(
            select(Invoice)
            .join(Subscription)
            .where(Subscription.user_id == user_id)
            .order_by(Invoice.created_at.desc())
            .limit(limit)
            .offset(offset),
        )
        return list(result.scalars().all())

    async def mark_invoice_paid(
        self,
        invoice_id: UUID,
        payment_method: str,
        payment_reference: str | None = None,
    ) -> Invoice:
        """Mark an invoice as paid.

        Args:
            invoice_id: Invoice ID
            payment_method: Payment method used
            payment_reference: Payment reference/transaction ID

        Returns:
            Updated invoice
        """
        result = await self.db.execute(
            select(Invoice).where(Invoice.id == invoice_id),
        )
        invoice = result.scalar_one_or_none()

        if invoice is None:
            raise ValueError(f"Invoice {invoice_id} not found")

        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = datetime.now(UTC)
        invoice.payment_method = payment_method
        invoice.payment_reference = payment_reference

        await self.db.flush()
        await self.db.refresh(invoice)

        self.logger.info(
            "invoice_paid",
            invoice_id=str(invoice_id),
            payment_method=payment_method,
        )

        return invoice

    async def update_subscription_status(self) -> int:
        """Update subscription statuses based on expiration.

        This should be called periodically by a scheduled task.

        Returns:
            Number of subscriptions updated
        """
        now = datetime.now(UTC)

        result = await self.db.execute(
            select(Subscription).where(
                Subscription.status == SubscriptionStatus.ACTIVE,
                Subscription.expires_at.isnot(None),
                Subscription.expires_at < now,
            ),
        )
        expired_subscriptions = result.scalars().all()

        count = 0
        for subscription in expired_subscriptions:
            subscription.status = SubscriptionStatus.EXPIRED
            count += 1
            self.logger.info(
                "subscription_expired",
                subscription_id=str(subscription.id),
                user_id=str(subscription.user_id),
            )

        if count > 0:
            await self.db.flush()

        return count

    def _calculate_expiration_date(
        self,
        start_date: datetime,
        billing_cycle: BillingCycle,
    ) -> datetime:
        """Calculate subscription expiration date.

        Args:
            start_date: Subscription start date
            billing_cycle: Billing cycle

        Returns:
            Expiration date
        """
        if billing_cycle == BillingCycle.MONTHLY:
            return start_date + timedelta(days=30)
        if billing_cycle == BillingCycle.YEARLY:
            return start_date + timedelta(days=365)
        raise ValueError(f"Invalid billing cycle: {billing_cycle}")
