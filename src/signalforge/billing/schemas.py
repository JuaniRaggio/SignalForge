"""Pydantic schemas for billing and subscription API."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from signalforge.billing.models import (
    BillingCycle,
    Currency,
    InvoiceStatus,
    SubscriptionStatus,
)
from signalforge.billing.tier_features import SubscriptionTier


class SubscriptionPlanBase(BaseModel):
    """Base schema for subscription plan."""

    name: str = Field(..., min_length=1, max_length=100)
    tier: SubscriptionTier
    description: str | None = Field(None, max_length=500)
    price_monthly: Decimal = Field(..., ge=0)
    price_yearly: Decimal = Field(..., ge=0)
    currency: Currency = Currency.USD
    features: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    trial_days: int = Field(0, ge=0)


class SubscriptionPlanCreate(SubscriptionPlanBase):
    """Schema for creating a subscription plan."""



class SubscriptionPlanUpdate(BaseModel):
    """Schema for updating a subscription plan."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    price_monthly: Decimal | None = Field(None, ge=0)
    price_yearly: Decimal | None = Field(None, ge=0)
    features: dict[str, Any] | None = None
    is_active: bool | None = None
    trial_days: int | None = Field(None, ge=0)


class SubscriptionPlanResponse(SubscriptionPlanBase):
    """Schema for subscription plan response."""

    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SubscriptionBase(BaseModel):
    """Base schema for subscription."""

    plan_id: UUID
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    auto_renew: bool = True


class SubscriptionCreate(SubscriptionBase):
    """Schema for creating a subscription."""



class SubscriptionUpdate(BaseModel):
    """Schema for updating a subscription."""

    plan_id: UUID | None = None
    billing_cycle: BillingCycle | None = None
    auto_renew: bool | None = None


class SubscriptionResponse(BaseModel):
    """Schema for subscription response."""

    id: UUID
    user_id: UUID
    plan_id: UUID
    status: SubscriptionStatus
    billing_cycle: BillingCycle
    started_at: datetime
    expires_at: datetime | None
    canceled_at: datetime | None
    trial_ends_at: datetime | None
    auto_renew: bool
    created_at: datetime
    updated_at: datetime
    plan: SubscriptionPlanResponse

    model_config = {"from_attributes": True}


class SubscriptionCancelRequest(BaseModel):
    """Schema for canceling a subscription."""

    cancel_immediately: bool = Field(
        False,
        description="If True, cancel immediately. If False, cancel at period end.",
    )
    reason: str | None = Field(None, max_length=500)


class InvoiceBase(BaseModel):
    """Base schema for invoice."""

    subscription_id: UUID
    invoice_number: str
    amount: Decimal = Field(..., ge=0)
    currency: Currency = Currency.USD
    billing_period_start: datetime
    billing_period_end: datetime
    due_date: datetime
    notes: str | None = Field(None, max_length=1000)


class InvoiceCreate(InvoiceBase):
    """Schema for creating an invoice."""



class InvoiceUpdate(BaseModel):
    """Schema for updating an invoice."""

    status: InvoiceStatus | None = None
    paid_at: datetime | None = None
    payment_method: str | None = Field(None, max_length=50)
    payment_reference: str | None = Field(None, max_length=255)
    notes: str | None = Field(None, max_length=1000)


class InvoiceResponse(InvoiceBase):
    """Schema for invoice response."""

    id: UUID
    status: InvoiceStatus
    paid_at: datetime | None
    payment_method: str | None
    payment_reference: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UsageRecordBase(BaseModel):
    """Base schema for usage record."""

    resource_type: str
    count: int = Field(1, ge=1)


class UsageRecordCreate(UsageRecordBase):
    """Schema for creating a usage record."""



class UsageRecordResponse(UsageRecordBase):
    """Schema for usage record response."""

    id: UUID
    user_id: UUID
    recorded_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UsageSummary(BaseModel):
    """Schema for usage summary."""

    resource_type: str
    total_usage: int
    quota_limit: int
    remaining: int
    is_unlimited: bool
    reset_at: datetime | None = None


class UserUsageResponse(BaseModel):
    """Schema for user usage response."""

    user_id: UUID
    tier: SubscriptionTier
    usage: list[UsageSummary]


class QuotaCheckRequest(BaseModel):
    """Schema for quota check request."""

    resource_type: str


class QuotaCheckResponse(BaseModel):
    """Schema for quota check response."""

    allowed: bool
    resource_type: str
    current_usage: int
    quota_limit: int
    remaining: int
    is_unlimited: bool
    message: str | None = None
