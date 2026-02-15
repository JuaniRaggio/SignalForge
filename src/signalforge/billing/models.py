"""Billing and subscription models."""

import enum
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from signalforge.billing.tier_features import SubscriptionTier
from signalforge.models.base import Base, TimestampMixin


class BillingCycle(str, enum.Enum):
    """Billing cycle enum."""

    MONTHLY = "monthly"
    YEARLY = "yearly"


class SubscriptionStatus(str, enum.Enum):
    """Subscription status enum."""

    ACTIVE = "active"
    CANCELED = "canceled"
    EXPIRED = "expired"
    PAST_DUE = "past_due"
    TRIALING = "trialing"


class InvoiceStatus(str, enum.Enum):
    """Invoice status enum."""

    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"
    VOIDED = "voided"


class Currency(str, enum.Enum):
    """Currency enum."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class SubscriptionPlan(Base, TimestampMixin):
    """Subscription plan model defining available tiers and pricing."""

    __tablename__ = "subscription_plans"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    tier: Mapped[SubscriptionTier] = mapped_column(
        Enum(SubscriptionTier, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        unique=True,
        index=True,
    )
    description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    price_monthly: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )
    price_yearly: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )
    currency: Mapped[Currency] = mapped_column(
        Enum(Currency, values_callable=lambda x: [e.value for e in x]),
        default=Currency.USD,
        nullable=False,
    )
    features: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )
    trial_days: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )

    # Relationships
    subscriptions: Mapped[list["Subscription"]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
    )


class Subscription(Base, TimestampMixin):
    """User subscription to a plan."""

    __tablename__ = "subscriptions"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("subscription_plans.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus, values_callable=lambda x: [e.value for e in x]),
        default=SubscriptionStatus.ACTIVE,
        nullable=False,
        index=True,
    )
    billing_cycle: Mapped[BillingCycle] = mapped_column(
        Enum(BillingCycle, values_callable=lambda x: [e.value for e in x]),
        default=BillingCycle.MONTHLY,
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    canceled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    trial_ends_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    auto_renew: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        default=dict,
        nullable=True,
    )

    # Relationships
    plan: Mapped["SubscriptionPlan"] = relationship(back_populates="subscriptions")
    invoices: Mapped[list["Invoice"]] = relationship(
        back_populates="subscription",
        cascade="all, delete-orphan",
    )

    @property
    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        now = datetime.now(UTC)
        return (
            self.status == SubscriptionStatus.ACTIVE
            and (self.expires_at is None or self.expires_at > now)
        )

    @property
    def is_trial(self) -> bool:
        """Check if subscription is in trial period."""
        now = datetime.now(UTC)
        return (
            self.status == SubscriptionStatus.TRIALING
            and self.trial_ends_at is not None
            and self.trial_ends_at > now
        )

    @property
    def days_until_expiration(self) -> int | None:
        """Get number of days until subscription expires."""
        if self.expires_at is None:
            return None
        now = datetime.now(UTC)
        delta = self.expires_at - now
        return max(0, delta.days)


class Invoice(Base, TimestampMixin):
    """Invoice for subscription billing."""

    __tablename__ = "invoices"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    subscription_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("subscriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    invoice_number: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )
    currency: Mapped[Currency] = mapped_column(
        Enum(Currency, values_callable=lambda x: [e.value for e in x]),
        default=Currency.USD,
        nullable=False,
    )
    status: Mapped[InvoiceStatus] = mapped_column(
        Enum(InvoiceStatus, values_callable=lambda x: [e.value for e in x]),
        default=InvoiceStatus.DRAFT,
        nullable=False,
        index=True,
    )
    billing_period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    billing_period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    due_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    payment_method: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )
    payment_reference: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        default=dict,
        nullable=True,
    )

    # Relationships
    subscription: Mapped["Subscription"] = relationship(back_populates="invoices")

    @property
    def is_paid(self) -> bool:
        """Check if invoice has been paid."""
        return self.status == InvoiceStatus.PAID

    @property
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        now = datetime.now(UTC)
        return (
            self.status in [InvoiceStatus.PENDING, InvoiceStatus.DRAFT]
            and self.due_date < now
        )

    @property
    def days_overdue(self) -> int:
        """Get number of days invoice is overdue."""
        if not self.is_overdue:
            return 0
        now = datetime.now(UTC)
        delta = now - self.due_date
        return delta.days


class UsageRecord(Base, TimestampMixin):
    """Record of resource usage for quota tracking."""

    __tablename__ = "usage_records"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resource_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    count: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        default=dict,
        nullable=True,
    )
