"""Tier feature definitions and configuration."""

import enum
from typing import Any


class SubscriptionTier(str, enum.Enum):
    """Subscription tier enum."""

    FREE = "free"
    PROSUMER = "prosumer"
    PROFESSIONAL = "professional"


class ResourceType(str, enum.Enum):
    """Resource types for quota management."""

    PREDICTIONS = "predictions"
    API_CALLS = "api_calls"
    HISTORY_ACCESS = "history_access"
    NLP_SUMMARIES = "nlp_summaries"
    SECTOR_REPORTS = "sector_reports"
    BULK_API = "bulk_api"
    WHITE_LABEL = "white_label"


TIER_FEATURES: dict[str, dict[str, Any]] = {
    "free": {
        "predictions_per_day": 10,
        "api_calls_per_minute": 5,
        "history_days": 30,
        "nlp_summaries": False,
        "sector_reports": False,
        "bulk_api": False,
        "white_label": False,
        "max_watchlist_symbols": 10,
        "real_time_alerts": False,
        "custom_models": False,
        "priority_support": False,
    },
    "prosumer": {
        "predictions_per_day": 100,
        "api_calls_per_minute": 60,
        "history_days": 365,
        "nlp_summaries": True,
        "sector_reports": True,
        "bulk_api": False,
        "white_label": False,
        "max_watchlist_symbols": 50,
        "real_time_alerts": True,
        "custom_models": False,
        "priority_support": True,
    },
    "professional": {
        "predictions_per_day": -1,  # unlimited
        "api_calls_per_minute": 300,
        "history_days": -1,  # unlimited
        "nlp_summaries": True,
        "sector_reports": True,
        "bulk_api": True,
        "white_label": True,
        "max_watchlist_symbols": -1,  # unlimited
        "real_time_alerts": True,
        "custom_models": True,
        "priority_support": True,
    },
}


def get_tier_feature(tier: SubscriptionTier, feature_name: str) -> Any:
    """Get a specific feature value for a tier.

    Args:
        tier: The subscription tier
        feature_name: Name of the feature to retrieve

    Returns:
        The feature value for the given tier

    Raises:
        KeyError: If tier or feature name is invalid
    """
    tier_config = TIER_FEATURES.get(tier.value)
    if tier_config is None:
        raise KeyError(f"Invalid tier: {tier}")

    if feature_name not in tier_config:
        raise KeyError(f"Invalid feature name: {feature_name}")

    return tier_config[feature_name]


def is_feature_enabled(tier: SubscriptionTier, feature_name: str) -> bool:
    """Check if a feature is enabled for a tier.

    Args:
        tier: The subscription tier
        feature_name: Name of the feature to check

    Returns:
        True if feature is enabled, False otherwise
    """
    try:
        value = get_tier_feature(tier, feature_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        return True
    except KeyError:
        return False


def get_quota_limit(tier: SubscriptionTier, resource: str) -> int:
    """Get the quota limit for a resource on a tier.

    Args:
        tier: The subscription tier
        resource: The resource type (e.g., 'predictions_per_day')

    Returns:
        The quota limit (-1 for unlimited, 0 for disabled, positive for limit)

    Raises:
        KeyError: If tier or resource is invalid
    """
    return int(get_tier_feature(tier, resource))


def has_unlimited_quota(tier: SubscriptionTier, resource: str) -> bool:
    """Check if a tier has unlimited quota for a resource.

    Args:
        tier: The subscription tier
        resource: The resource type

    Returns:
        True if unlimited, False otherwise
    """
    try:
        limit = get_quota_limit(tier, resource)
        return limit == -1
    except (KeyError, TypeError):
        return False
