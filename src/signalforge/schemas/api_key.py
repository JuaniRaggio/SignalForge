"""API Key schemas for requests and responses."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from signalforge.models.api_key import SubscriptionTier


class APIKeyCreate(BaseModel):
    """Schema for creating a new API key."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Descriptive name for the API key",
    )
    tier: SubscriptionTier = Field(
        default=SubscriptionTier.FREE,
        description="Subscription tier for the key",
    )
    scopes: list[str] = Field(
        default_factory=lambda: ["read", "write"],
        description="List of allowed scopes",
    )
    rate_limit_override: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description="Custom rate limit (overrides tier default)",
    )
    burst_limit_override: int | None = Field(
        default=None,
        ge=1,
        le=20000,
        description="Custom burst limit (overrides tier default)",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiration datetime",
    )


class APIKeyResponse(BaseModel):
    """Schema for API key response."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID = Field(
        ...,
        description="API key ID",
    )
    user_id: UUID = Field(
        ...,
        description="User ID that owns the key",
    )
    name: str = Field(
        ...,
        description="Descriptive name for the key",
    )
    tier: SubscriptionTier = Field(
        ...,
        description="Subscription tier",
    )
    scopes: list[str] = Field(
        ...,
        description="List of allowed scopes",
    )
    rate_limit_override: int | None = Field(
        default=None,
        description="Custom rate limit",
    )
    burst_limit_override: int | None = Field(
        default=None,
        description="Custom burst limit",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Expiration datetime",
    )
    last_used_at: datetime | None = Field(
        default=None,
        description="Last usage timestamp",
    )
    is_active: bool = Field(
        ...,
        description="Whether the key is active",
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        ...,
        description="Last update timestamp",
    )


class APIKeyCreatedResponse(BaseModel):
    """Schema for newly created API key with plain key."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(
        ...,
        description="The plain API key (shown only once)",
    )
    key_info: APIKeyResponse = Field(
        ...,
        description="API key metadata",
    )
    message: str = Field(
        default="API key created successfully. Store this key securely as it will not be shown again.",
        description="Warning message",
    )


class APIKeyUpdate(BaseModel):
    """Schema for updating an API key."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New name for the key",
    )
    scopes: list[str] | None = Field(
        default=None,
        description="Updated scopes",
    )
    is_active: bool | None = Field(
        default=None,
        description="Active status",
    )


class UsageStatsResponse(BaseModel):
    """Schema for API key usage statistics."""

    model_config = ConfigDict(extra="forbid")

    user_id: UUID = Field(
        ...,
        description="User ID",
    )
    period: str = Field(
        ...,
        description="Time period (today, week, month)",
    )
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total number of requests",
    )
    successful_requests: int = Field(
        ...,
        ge=0,
        description="Number of successful requests",
    )
    failed_requests: int = Field(
        ...,
        ge=0,
        description="Number of failed requests",
    )
    rate_limited_requests: int = Field(
        ...,
        ge=0,
        description="Number of rate-limited requests",
    )
    average_response_time_ms: float = Field(
        ...,
        ge=0,
        description="Average response time in milliseconds",
    )
    endpoints_used: dict[str, int] = Field(
        ...,
        description="Breakdown of requests by endpoint",
    )
    peak_hour: str | None = Field(
        default=None,
        description="Hour with peak usage",
    )
    total_data_transferred_bytes: int = Field(
        ...,
        ge=0,
        description="Total data transferred in bytes",
    )


class RateLimitStatusResponse(BaseModel):
    """Schema for rate limit status."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        ...,
        description="Rate limit (requests per minute)",
    )
    remaining: int = Field(
        ...,
        ge=0,
        description="Remaining requests in current window",
    )
    used: int = Field(
        ...,
        ge=0,
        description="Requests used in current window",
    )
    reset_in_seconds: int = Field(
        ...,
        ge=0,
        description="Seconds until rate limit resets",
    )
    is_burst: bool = Field(
        ...,
        description="Whether this is burst limit status",
    )
    tier: str = Field(
        ...,
        description="Subscription tier",
    )
