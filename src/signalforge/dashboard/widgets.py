"""Dashboard widgets for SignalForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.dashboard.schemas import WidgetType

logger = structlog.get_logger(__name__)


class BaseWidget(ABC):
    """Base class for dashboard widgets."""

    widget_type: WidgetType

    def __init__(self, session: AsyncSession | None = None) -> None:
        """Initialize widget with optional database session."""
        self.session = session

    @abstractmethod
    async def get_data(
        self, user_id: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Get widget data.

        Args:
            user_id: User identifier
            settings: Widget-specific settings

        Returns:
            Dictionary containing widget data
        """
        ...

    @abstractmethod
    def get_default_settings(self) -> dict[str, Any]:
        """Get default widget settings.

        Returns:
            Dictionary of default settings
        """
        ...

    async def validate_settings(self, _settings: dict[str, Any]) -> bool:
        """Validate widget settings.

        Args:
            _settings: Settings to validate (unused in base class)

        Returns:
            True if settings are valid
        """
        return True


class PortfolioSummaryWidget(BaseWidget):
    """Portfolio summary widget showing total value and P&L."""

    widget_type = WidgetType.PORTFOLIO_SUMMARY

    async def get_data(
        self, user_id: str, _settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return portfolio summary data.

        Returns:
            - total_value: Current portfolio value
            - daily_pnl: Daily profit/loss
            - daily_pnl_pct: Daily P&L percentage
            - total_pnl: Total profit/loss
            - total_pnl_pct: Total P&L percentage
            - top_gainers: List of top gaining positions
            - top_losers: List of top losing positions
        """
        logger.info("portfolio_summary_widget.get_data", user_id=user_id)

        # TODO: Implement actual portfolio data fetching from database
        # This is a placeholder implementation
        return {
            "total_value": 100000.00,
            "daily_pnl": 1250.50,
            "daily_pnl_pct": 1.25,
            "total_pnl": 15000.00,
            "total_pnl_pct": 15.0,
            "top_gainers": [
                {"symbol": "AAPL", "pnl": 500.00, "pnl_pct": 5.0},
                {"symbol": "MSFT", "pnl": 400.00, "pnl_pct": 4.5},
            ],
            "top_losers": [
                {"symbol": "TSLA", "pnl": -200.00, "pnl_pct": -2.0},
            ],
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for portfolio summary widget."""
        return {
            "show_gainers_losers": True,
            "num_gainers_losers": 3,
            "include_cash": True,
        }


class WatchlistWidget(BaseWidget):
    """Watchlist widget showing tracked symbols with current prices."""

    widget_type = WidgetType.WATCHLIST

    async def get_data(
        self, user_id: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return watchlist with current prices and changes.

        Returns:
            - symbols: List of watchlist items with price data
        """
        logger.info("watchlist_widget.get_data", user_id=user_id)

        watchlist_name = settings.get("watchlist_name", "default")

        # TODO: Fetch actual watchlist from database
        return {
            "watchlist_name": watchlist_name,
            "symbols": [
                {
                    "symbol": "AAPL",
                    "price": 185.50,
                    "change": 2.50,
                    "change_pct": 1.37,
                    "volume": 52000000,
                },
                {
                    "symbol": "MSFT",
                    "price": 420.25,
                    "change": -1.25,
                    "change_pct": -0.30,
                    "volume": 28000000,
                },
            ],
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for watchlist widget."""
        return {
            "watchlist_name": "default",
            "show_volume": True,
            "show_change_pct": True,
        }


class SignalsFeedWidget(BaseWidget):
    """Signals feed widget showing recent trading signals."""

    widget_type = WidgetType.SIGNALS_FEED

    async def get_data(
        self, user_id: str, _settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return recent signals for user's symbols.

        Returns:
            - signals: List of recent signals
        """
        logger.info("signals_feed_widget.get_data", user_id=user_id)

        # TODO: Fetch actual signals from database
        # Use settings["limit"] and settings["signal_types"] when implemented
        return {
            "signals": [
                {
                    "signal_id": "sig_001",
                    "symbol": "AAPL",
                    "signal_type": "buy",
                    "strength": 0.85,
                    "price": 185.50,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "ml_model",
                },
                {
                    "signal_id": "sig_002",
                    "symbol": "MSFT",
                    "signal_type": "sell",
                    "strength": 0.72,
                    "price": 420.25,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "technical_analysis",
                },
            ],
            "total_count": 2,
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for signals feed widget."""
        return {
            "limit": 10,
            "signal_types": ["all"],
            "min_strength": 0.5,
        }


class MarketOverviewWidget(BaseWidget):
    """Market overview widget showing major indices and sentiment."""

    widget_type = WidgetType.MARKET_OVERVIEW

    async def get_data(
        self, user_id: str, _settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return market indices and overall sentiment.

        Returns:
            - indices: Market indices data
            - sentiment: Overall market sentiment
        """
        logger.info("market_overview_widget.get_data", user_id=user_id)

        # TODO: Fetch actual market data
        return {
            "indices": [
                {
                    "symbol": "^GSPC",
                    "name": "S&P 500",
                    "value": 5000.00,
                    "change": 25.50,
                    "change_pct": 0.51,
                },
                {
                    "symbol": "^IXIC",
                    "name": "NASDAQ",
                    "value": 16000.00,
                    "change": -30.25,
                    "change_pct": -0.19,
                },
                {
                    "symbol": "^DJI",
                    "name": "Dow Jones",
                    "value": 38000.00,
                    "change": 150.00,
                    "change_pct": 0.40,
                },
            ],
            "sentiment": {
                "score": 0.65,
                "label": "bullish",
                "vix": 15.5,
            },
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for market overview widget."""
        return {
            "indices": ["^GSPC", "^IXIC", "^DJI"],
            "show_vix": True,
        }


class SectorHeatmapWidget(BaseWidget):
    """Sector heatmap widget showing sector performance."""

    widget_type = WidgetType.SECTOR_HEATMAP

    async def get_data(
        self, user_id: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return sector performance heatmap data.

        Returns:
            - sectors: List of sectors with performance data
        """
        logger.info("sector_heatmap_widget.get_data", user_id=user_id)

        timeframe = settings.get("timeframe", "1d")

        # TODO: Fetch actual sector data
        return {
            "timeframe": timeframe,
            "sectors": [
                {"name": "Technology", "change_pct": 1.25, "volume": 1500000000},
                {"name": "Healthcare", "change_pct": 0.75, "volume": 800000000},
                {"name": "Financials", "change_pct": -0.50, "volume": 1200000000},
                {"name": "Energy", "change_pct": 2.10, "volume": 600000000},
                {"name": "Consumer", "change_pct": -1.20, "volume": 900000000},
            ],
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for sector heatmap widget."""
        return {
            "timeframe": "1d",
            "show_volume": True,
        }


class AlertsWidget(BaseWidget):
    """Alerts widget showing user alerts and notifications."""

    widget_type = WidgetType.ALERTS

    async def get_data(
        self, user_id: str, _settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return user alerts.

        Returns:
            - alerts: List of alerts
        """
        logger.info("alerts_widget.get_data", user_id=user_id)

        # TODO: Fetch actual alerts from database
        return {
            "alerts": [
                {
                    "alert_id": "alert_001",
                    "type": "price",
                    "symbol": "AAPL",
                    "message": "AAPL reached target price of $185.00",
                    "severity": "info",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            ],
            "unread_count": 1,
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for alerts widget."""
        return {
            "show_read": False,
            "severity_filter": ["info", "warning", "error"],
        }


class PerformanceChartWidget(BaseWidget):
    """Performance chart widget showing portfolio performance over time."""

    widget_type = WidgetType.PERFORMANCE_CHART

    async def get_data(
        self, user_id: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return performance chart data.

        Returns:
            - data_points: Time series data
        """
        logger.info("performance_chart_widget.get_data", user_id=user_id)

        timeframe = settings.get("timeframe", "1M")

        # TODO: Fetch actual performance data
        return {
            "timeframe": timeframe,
            "data_points": [
                {"timestamp": "2026-01-01", "value": 95000.00},
                {"timestamp": "2026-01-15", "value": 98000.00},
                {"timestamp": "2026-02-01", "value": 100000.00},
            ],
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for performance chart widget."""
        return {
            "timeframe": "1M",
            "show_benchmark": True,
            "benchmark": "^GSPC",
        }


class NewsFeedWidget(BaseWidget):
    """News feed widget showing recent market news."""

    widget_type = WidgetType.NEWS_FEED

    async def get_data(
        self, user_id: str, _settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return news feed.

        Returns:
            - articles: List of news articles
        """
        logger.info("news_feed_widget.get_data", user_id=user_id)

        # TODO: Fetch actual news from database
        # Use settings["limit"] when implemented
        return {
            "articles": [
                {
                    "article_id": "news_001",
                    "title": "Market reaches new highs",
                    "source": "Financial Times",
                    "timestamp": datetime.utcnow().isoformat(),
                    "sentiment": 0.75,
                    "url": "https://example.com/article",
                },
            ],
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for news feed widget."""
        return {
            "limit": 10,
            "sources": ["all"],
            "sentiment_filter": ["positive", "neutral", "negative"],
        }


class PositionsWidget(BaseWidget):
    """Positions widget showing current positions."""

    widget_type = WidgetType.POSITIONS

    async def get_data(
        self, user_id: str, _settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return current positions.

        Returns:
            - positions: List of positions
        """
        logger.info("positions_widget.get_data", user_id=user_id)

        # TODO: Fetch actual positions from database
        return {
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "avg_price": 180.00,
                    "current_price": 185.50,
                    "pnl": 550.00,
                    "pnl_pct": 3.06,
                },
            ],
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for positions widget."""
        return {
            "show_closed": False,
            "sort_by": "pnl_pct",
        }


class PredictionsWidget(BaseWidget):
    """Predictions widget showing ML model predictions."""

    widget_type = WidgetType.PREDICTIONS

    async def get_data(
        self, user_id: str, _settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Return predictions.

        Returns:
            - predictions: List of predictions
        """
        logger.info("predictions_widget.get_data", user_id=user_id)

        # TODO: Fetch actual predictions from ML models
        return {
            "predictions": [
                {
                    "symbol": "AAPL",
                    "predicted_price": 190.00,
                    "confidence": 0.82,
                    "horizon": "1d",
                    "model": "lstm_v1",
                },
            ],
        }

    def get_default_settings(self) -> dict[str, Any]:
        """Get default settings for predictions widget."""
        return {
            "horizon": "1d",
            "min_confidence": 0.7,
        }


class WidgetFactory:
    """Factory for creating widget instances."""

    _widgets: dict[WidgetType, type[BaseWidget]] = {}

    @classmethod
    def register(cls, widget_class: type[BaseWidget]) -> None:
        """Register a widget class.

        Args:
            widget_class: Widget class to register
        """
        if not hasattr(widget_class, "widget_type"):
            msg = f"Widget class {widget_class.__name__} must have widget_type attribute"
            raise ValueError(msg)

        cls._widgets[widget_class.widget_type] = widget_class
        logger.debug(
            "widget_factory.register",
            widget_type=widget_class.widget_type.value,
            widget_class=widget_class.__name__,
        )

    @classmethod
    def create(
        cls, widget_type: WidgetType, session: AsyncSession | None = None
    ) -> BaseWidget:
        """Create widget instance.

        Args:
            widget_type: Type of widget to create
            session: Optional database session

        Returns:
            Widget instance

        Raises:
            ValueError: If widget type is not registered
        """
        widget_class = cls._widgets.get(widget_type)
        if widget_class is None:
            msg = f"Widget type {widget_type} is not registered"
            raise ValueError(msg)

        return widget_class(session=session)

    @classmethod
    def get_available_widgets(cls) -> list[WidgetType]:
        """Get list of available widget types.

        Returns:
            List of registered widget types
        """
        return list(cls._widgets.keys())


# Register all widgets
WidgetFactory.register(PortfolioSummaryWidget)
WidgetFactory.register(WatchlistWidget)
WidgetFactory.register(SignalsFeedWidget)
WidgetFactory.register(MarketOverviewWidget)
WidgetFactory.register(SectorHeatmapWidget)
WidgetFactory.register(AlertsWidget)
WidgetFactory.register(PerformanceChartWidget)
WidgetFactory.register(NewsFeedWidget)
WidgetFactory.register(PositionsWidget)
WidgetFactory.register(PredictionsWidget)
