"""Dashboard module for SignalForge."""

from signalforge.dashboard.config import DashboardConfigManager
from signalforge.dashboard.schemas import (
    DashboardLayout,
    StreamMessage,
    WidgetConfig,
    WidgetData,
    WidgetSize,
    WidgetType,
)
from signalforge.dashboard.streaming import ConnectionManager, StreamingService
from signalforge.dashboard.widgets import (
    AlertsWidget,
    BaseWidget,
    MarketOverviewWidget,
    NewsFeedWidget,
    PerformanceChartWidget,
    PortfolioSummaryWidget,
    PositionsWidget,
    PredictionsWidget,
    SectorHeatmapWidget,
    SignalsFeedWidget,
    WatchlistWidget,
    WidgetFactory,
)

__all__ = [
    # Schemas
    "DashboardLayout",
    "StreamMessage",
    "WidgetConfig",
    "WidgetData",
    "WidgetSize",
    "WidgetType",
    # Config
    "DashboardConfigManager",
    # Streaming
    "ConnectionManager",
    "StreamingService",
    # Widgets
    "AlertsWidget",
    "BaseWidget",
    "MarketOverviewWidget",
    "NewsFeedWidget",
    "PerformanceChartWidget",
    "PortfolioSummaryWidget",
    "PositionsWidget",
    "PredictionsWidget",
    "SectorHeatmapWidget",
    "SignalsFeedWidget",
    "WatchlistWidget",
    "WidgetFactory",
]
