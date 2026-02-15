"""Alert message templates."""

from __future__ import annotations

from signalforge.alerts.schemas import AlertPriority, AlertType


class AlertTemplates:
    """Alert message templates."""

    TEMPLATES = {
        "signal_bullish": "{symbol}: Bullish signal detected! {confidence}% confidence. {reason}",
        "signal_bearish": "{symbol}: Bearish signal detected. {confidence}% confidence. {reason}",
        "price_target_hit": "{symbol} hit your price target of ${target}! Current: ${current}",
        "price_target_approaching": "{symbol} approaching your price target of ${target}. Current: ${current}",
        "earnings_upcoming": "{symbol} reports earnings in {days} days. Analyst consensus: {consensus}",
        "earnings_today": "{symbol} reports earnings today! Analyst consensus: {consensus}",
        "portfolio_gain": "Your {symbol} position is up {gain}% today! Current value: ${value}",
        "portfolio_loss": "Alert: {symbol} is down {loss}% today. Consider reviewing your position.",
        "portfolio_rebalance": "Portfolio rebalance suggested: {reason}",
        "watchlist_move": "{symbol} moved {direction} {percent}% - now at ${price}",
        "watchlist_breakout": "{symbol} breakout detected! Price: ${price}, Volume: {volume}",
        "news_positive": "{symbol}: Positive news detected - {headline}",
        "news_negative": "{symbol}: Negative news detected - {headline}",
        "system_maintenance": "System maintenance scheduled: {message}",
        "system_error": "System alert: {message}",
    }

    TITLES = {
        (AlertType.SIGNAL, AlertPriority.LOW): "Signal Detected",
        (AlertType.SIGNAL, AlertPriority.MEDIUM): "Signal Alert",
        (AlertType.SIGNAL, AlertPriority.HIGH): "Strong Signal Alert",
        (AlertType.SIGNAL, AlertPriority.CRITICAL): "CRITICAL Signal",
        (AlertType.PRICE_TARGET, AlertPriority.LOW): "Price Update",
        (AlertType.PRICE_TARGET, AlertPriority.MEDIUM): "Price Target Alert",
        (AlertType.PRICE_TARGET, AlertPriority.HIGH): "Price Target Reached",
        (AlertType.PRICE_TARGET, AlertPriority.CRITICAL): "CRITICAL Price Movement",
        (AlertType.EARNINGS, AlertPriority.LOW): "Earnings Notice",
        (AlertType.EARNINGS, AlertPriority.MEDIUM): "Earnings Alert",
        (AlertType.EARNINGS, AlertPriority.HIGH): "Earnings Today",
        (AlertType.EARNINGS, AlertPriority.CRITICAL): "CRITICAL Earnings Event",
        (AlertType.NEWS, AlertPriority.LOW): "News Update",
        (AlertType.NEWS, AlertPriority.MEDIUM): "News Alert",
        (AlertType.NEWS, AlertPriority.HIGH): "Important News",
        (AlertType.NEWS, AlertPriority.CRITICAL): "BREAKING NEWS",
        (AlertType.PORTFOLIO, AlertPriority.LOW): "Portfolio Update",
        (AlertType.PORTFOLIO, AlertPriority.MEDIUM): "Portfolio Alert",
        (AlertType.PORTFOLIO, AlertPriority.HIGH): "Portfolio Action Required",
        (AlertType.PORTFOLIO, AlertPriority.CRITICAL): "CRITICAL Portfolio Alert",
        (AlertType.WATCHLIST, AlertPriority.LOW): "Watchlist Update",
        (AlertType.WATCHLIST, AlertPriority.MEDIUM): "Watchlist Alert",
        (AlertType.WATCHLIST, AlertPriority.HIGH): "Watchlist Movement",
        (AlertType.WATCHLIST, AlertPriority.CRITICAL): "CRITICAL Watchlist Alert",
        (AlertType.SYSTEM, AlertPriority.LOW): "System Notice",
        (AlertType.SYSTEM, AlertPriority.MEDIUM): "System Alert",
        (AlertType.SYSTEM, AlertPriority.HIGH): "System Warning",
        (AlertType.SYSTEM, AlertPriority.CRITICAL): "CRITICAL System Alert",
    }

    def render(self, template_name: str, **kwargs: str | int | float) -> str:
        """Render a template with provided values.

        Args:
            template_name: Name of template to render
            **kwargs: Template variables

        Returns:
            Rendered message

        Raises:
            KeyError: If template not found
        """
        template = self.TEMPLATES.get(template_name)
        if not template:
            raise KeyError(f"Template '{template_name}' not found")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}") from e

    def get_title(self, alert_type: AlertType, priority: AlertPriority) -> str:
        """Get appropriate title for alert type and priority.

        Args:
            alert_type: Type of alert
            priority: Alert priority

        Returns:
            Title string
        """
        return self.TITLES.get(
            (alert_type, priority),
            f"{alert_type.value.title()} Alert",
        )

    def add_template(self, name: str, template: str) -> None:
        """Add a custom template.

        Args:
            name: Template name
            template: Template string with {var} placeholders
        """
        self.TEMPLATES[name] = template

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template names
        """
        return list(self.TEMPLATES.keys())

    def get_template(self, name: str) -> str | None:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template string or None if not found
        """
        return self.TEMPLATES.get(name)
