"""Dashboard schemas for SignalForge."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WidgetType(str, Enum):
    """Types of dashboard widgets."""

    PORTFOLIO_SUMMARY = "portfolio_summary"
    WATCHLIST = "watchlist"
    SIGNALS_FEED = "signals_feed"
    MARKET_OVERVIEW = "market_overview"
    SECTOR_HEATMAP = "sector_heatmap"
    ALERTS = "alerts"
    PERFORMANCE_CHART = "performance_chart"
    NEWS_FEED = "news_feed"
    POSITIONS = "positions"
    PREDICTIONS = "predictions"


class WidgetSize(str, Enum):
    """Widget size options."""

    SMALL = "small"  # 1x1
    MEDIUM = "medium"  # 2x1
    LARGE = "large"  # 2x2
    WIDE = "wide"  # 3x1
    TALL = "tall"  # 1x3


class WidgetConfig(BaseModel):
    """Configuration for a dashboard widget."""

    widget_id: str = Field(..., description="Unique identifier for the widget")
    widget_type: WidgetType = Field(..., description="Type of widget")
    size: WidgetSize = Field(..., description="Widget size")
    position: tuple[int, int] = Field(..., description="Position (row, col)")
    settings: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Widget-specific settings"
    )
    refresh_interval: int = Field(
        default=30, description="Refresh interval in seconds", ge=5, le=300
    )


class DashboardLayout(BaseModel):
    """User's dashboard layout configuration."""

    user_id: str = Field(..., description="User identifier")
    layout_name: str = Field(..., description="Layout name")
    widgets: list[WidgetConfig] = Field(
        default_factory=list, description="List of widgets in the layout"
    )
    columns: int = Field(default=4, description="Number of columns", ge=1, le=12)
    is_default: bool = Field(default=False, description="Is this the default layout")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


class WidgetData(BaseModel):
    """Data returned by a widget."""

    widget_id: str = Field(..., description="Widget identifier")
    widget_type: WidgetType = Field(..., description="Widget type")
    data: dict[str, Any] = Field(
        ..., description="Widget data payload"
    )
    last_updated: datetime = Field(..., description="Last update timestamp")
    next_update: datetime = Field(..., description="Next scheduled update")


class StreamMessage(BaseModel):
    """WebSocket stream message."""

    message_type: str = Field(
        ..., description="Message type: widget_update, alert, price, signal"
    )
    payload: dict[str, Any] = Field(
        ..., description="Message payload"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Message timestamp"
    )


class DashboardLayoutResponse(BaseModel):
    """Response model for dashboard layout."""

    layout: DashboardLayout
    status: str = "success"


class WidgetDataResponse(BaseModel):
    """Response model for widget data."""

    widget_data: WidgetData
    status: str = "success"


class WidgetConfigResponse(BaseModel):
    """Response model for widget configuration."""

    widget_config: WidgetConfig
    status: str = "success"


class SubscribeMessage(BaseModel):
    """WebSocket subscription message."""

    action: str = Field(..., description="Action: subscribe or unsubscribe")
    channels: list[str] = Field(..., description="Channels to subscribe/unsubscribe")


class WidgetUpdateMessage(BaseModel):
    """Widget update control message."""

    action: str = Field(..., description="Action: start or stop")
    widget_id: str = Field(..., description="Widget identifier")
