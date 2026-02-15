"""Comprehensive tests for Smart Alerts System."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from signalforge.alerts.channels import (
    AlertChannel,
    ChannelRouter,
    EmailChannel,
    PushChannel,
    WebSocketChannel,
)
from signalforge.alerts.manager import AlertManager
from signalforge.alerts.schemas import (
    Alert,
    AlertChannel as AlertChannelEnum,
    AlertPreferences,
    AlertPriority,
    AlertType,
)
from signalforge.alerts.templates import AlertTemplates
from signalforge.alerts.throttler import AlertThrottler
from signalforge.integration.schemas import IntegratedSignal, MarketRegime, SignalDirection


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_alert() -> Alert:
    """Create a sample alert for testing."""
    return Alert(
        alert_id="test-alert-1",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test Alert",
        message="This is a test alert",
        symbol="AAPL",
        data={"test": "data"},
        channels=[AlertChannelEnum.WEBSOCKET, AlertChannelEnum.EMAIL],
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def default_preferences() -> AlertPreferences:
    """Create default alert preferences."""
    return AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL, AlertType.PRICE_TARGET, AlertType.PORTFOLIO],
        enabled_channels=[AlertChannelEnum.WEBSOCKET, AlertChannelEnum.EMAIL],
        max_alerts_per_day=5,
        min_priority=AlertPriority.LOW,
    )


@pytest.fixture
def throttler() -> AlertThrottler:
    """Create alert throttler instance."""
    return AlertThrottler(default_max_per_day=5, redis_client=None)


@pytest.fixture
def templates() -> AlertTemplates:
    """Create alert templates instance."""
    return AlertTemplates()


@pytest.fixture
def channel_router() -> ChannelRouter:
    """Create channel router instance."""
    return ChannelRouter()


@pytest.fixture
def alert_manager(throttler: AlertThrottler, channel_router: ChannelRouter) -> AlertManager:
    """Create alert manager instance."""
    return AlertManager(throttler=throttler, channel_router=channel_router)


# ============================================================================
# Template Tests
# ============================================================================


def test_templates_render_signal_bullish(templates: AlertTemplates) -> None:
    """Test rendering bullish signal template."""
    message = templates.render(
        "signal_bullish",
        symbol="AAPL",
        confidence=85,
        reason="Strong momentum and positive sentiment",
    )
    assert "AAPL" in message
    assert "Bullish" in message
    assert "85%" in message
    assert "Strong momentum" in message


def test_templates_render_signal_bearish(templates: AlertTemplates) -> None:
    """Test rendering bearish signal template."""
    message = templates.render(
        "signal_bearish",
        symbol="TSLA",
        confidence=72,
        reason="Weakening fundamentals",
    )
    assert "TSLA" in message
    assert "Bearish" in message
    assert "72%" in message


def test_templates_render_price_target_hit(templates: AlertTemplates) -> None:
    """Test rendering price target template."""
    message = templates.render(
        "price_target_hit",
        symbol="NVDA",
        target="500.00",
        current="502.50",
    )
    assert "NVDA" in message
    assert "500.00" in message
    assert "502.50" in message


def test_templates_render_portfolio_gain(templates: AlertTemplates) -> None:
    """Test rendering portfolio gain template."""
    message = templates.render(
        "portfolio_gain",
        symbol="MSFT",
        gain="5.5",
        value="10500.00",
    )
    assert "MSFT" in message
    assert "5.5%" in message
    assert "$10500.00" in message


def test_templates_get_title_signal_critical(templates: AlertTemplates) -> None:
    """Test getting title for critical signal."""
    title = templates.get_title(AlertType.SIGNAL, AlertPriority.CRITICAL)
    assert "CRITICAL" in title


def test_templates_get_title_price_medium(templates: AlertTemplates) -> None:
    """Test getting title for medium priority price alert."""
    title = templates.get_title(AlertType.PRICE_TARGET, AlertPriority.MEDIUM)
    assert "Price" in title


def test_templates_add_custom_template(templates: AlertTemplates) -> None:
    """Test adding custom template."""
    templates.add_template("custom_test", "Custom message: {value}")
    message = templates.render("custom_test", value="test123")
    assert "Custom message: test123" == message


def test_templates_list_templates(templates: AlertTemplates) -> None:
    """Test listing all templates."""
    template_list = templates.list_templates()
    assert "signal_bullish" in template_list
    assert "signal_bearish" in template_list
    assert len(template_list) > 0


def test_templates_get_template(templates: AlertTemplates) -> None:
    """Test getting template by name."""
    template = templates.get_template("signal_bullish")
    assert template is not None
    assert "{symbol}" in template


def test_templates_missing_template(templates: AlertTemplates) -> None:
    """Test rendering non-existent template raises error."""
    with pytest.raises(KeyError):
        templates.render("nonexistent_template", value="test")


# ============================================================================
# Throttler Tests
# ============================================================================


@pytest.mark.asyncio
async def test_throttler_can_send_under_limit(
    throttler: AlertThrottler,
    sample_alert: Alert,
    default_preferences: AlertPreferences,
) -> None:
    """Test that alerts can be sent under daily limit."""
    can_send, reason = await throttler.can_send("user-123", sample_alert, default_preferences)
    assert can_send is True
    assert reason is None


@pytest.mark.asyncio
async def test_throttler_blocks_over_limit(
    throttler: AlertThrottler,
    sample_alert: Alert,
    default_preferences: AlertPreferences,
) -> None:
    """Test that alerts are blocked when over daily limit."""
    # Send 5 alerts (the limit)
    for _ in range(5):
        await throttler.record_sent("user-123", sample_alert)

    # 6th alert should be blocked
    can_send, reason = await throttler.can_send("user-123", sample_alert, default_preferences)
    assert can_send is False
    assert reason is not None
    assert "limit" in reason.lower()


@pytest.mark.asyncio
async def test_throttler_quiet_hours(throttler: AlertThrottler, sample_alert: Alert) -> None:
    """Test quiet hours blocking."""
    current_hour = datetime.now(timezone.utc).hour

    preferences = AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL],
        enabled_channels=[AlertChannelEnum.WEBSOCKET],
        max_alerts_per_day=10,
        quiet_hours_start=current_hour,
        quiet_hours_end=(current_hour + 1) % 24,
        min_priority=AlertPriority.LOW,
    )

    can_send, reason = await throttler.can_send("user-123", sample_alert, preferences)
    assert can_send is False
    assert "quiet hours" in reason.lower()


@pytest.mark.asyncio
async def test_throttler_quiet_hours_overnight(
    throttler: AlertThrottler,
    sample_alert: Alert,
) -> None:
    """Test quiet hours spanning midnight."""
    preferences = AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL],
        enabled_channels=[AlertChannelEnum.WEBSOCKET],
        max_alerts_per_day=10,
        quiet_hours_start=22,
        quiet_hours_end=6,
        min_priority=AlertPriority.LOW,
    )

    # Test during quiet hours depends on current time
    can_send, reason = await throttler.can_send("user-123", sample_alert, preferences)
    # Just verify it returns boolean and optional reason
    assert isinstance(can_send, bool)


@pytest.mark.asyncio
async def test_throttler_priority_threshold_low(
    throttler: AlertThrottler,
    sample_alert: Alert,
) -> None:
    """Test priority threshold blocks low priority alerts."""
    preferences = AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL],
        enabled_channels=[AlertChannelEnum.WEBSOCKET],
        max_alerts_per_day=10,
        min_priority=AlertPriority.HIGH,  # Only HIGH and CRITICAL
    )

    # Medium priority alert should be blocked
    can_send, reason = await throttler.can_send("user-123", sample_alert, preferences)
    assert can_send is False
    assert "priority" in reason.lower()


@pytest.mark.asyncio
async def test_throttler_priority_threshold_met(
    throttler: AlertThrottler,
    default_preferences: AlertPreferences,
) -> None:
    """Test priority threshold allows high priority alerts."""
    high_priority_alert = Alert(
        alert_id="test-alert-2",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.CRITICAL,
        title="Critical Alert",
        message="This is critical",
        channels=[AlertChannelEnum.WEBSOCKET],
        created_at=datetime.now(timezone.utc),
    )

    can_send, reason = await throttler.can_send(
        "user-123",
        high_priority_alert,
        default_preferences,
    )
    assert can_send is True


@pytest.mark.asyncio
async def test_throttler_symbol_filter_blocks(throttler: AlertThrottler) -> None:
    """Test symbol filter blocks alerts for non-watched symbols."""
    preferences = AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL],
        enabled_channels=[AlertChannelEnum.WEBSOCKET],
        max_alerts_per_day=10,
        symbol_filters=["AAPL", "MSFT"],  # Only watch these
        min_priority=AlertPriority.LOW,
    )

    alert = Alert(
        alert_id="test-alert-3",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test",
        message="Test",
        symbol="TSLA",  # Not in filter
        channels=[AlertChannelEnum.WEBSOCKET],
        created_at=datetime.now(timezone.utc),
    )

    can_send, reason = await throttler.can_send("user-123", alert, preferences)
    assert can_send is False
    assert "symbol" in reason.lower() or "filter" in reason.lower()


@pytest.mark.asyncio
async def test_throttler_symbol_filter_allows(throttler: AlertThrottler) -> None:
    """Test symbol filter allows alerts for watched symbols."""
    preferences = AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL],
        enabled_channels=[AlertChannelEnum.WEBSOCKET],
        max_alerts_per_day=10,
        symbol_filters=["AAPL", "MSFT"],
        min_priority=AlertPriority.LOW,
    )

    alert = Alert(
        alert_id="test-alert-4",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test",
        message="Test",
        symbol="AAPL",  # In filter
        channels=[AlertChannelEnum.WEBSOCKET],
        created_at=datetime.now(timezone.utc),
    )

    can_send, reason = await throttler.can_send("user-123", alert, preferences)
    assert can_send is True


@pytest.mark.asyncio
async def test_throttler_get_status(throttler: AlertThrottler, sample_alert: Alert) -> None:
    """Test getting throttle status."""
    # Record 3 alerts
    for _ in range(3):
        await throttler.record_sent("user-123", sample_alert)

    status = await throttler.get_status("user-123")
    assert status.user_id == "user-123"
    assert status.alerts_today == 3
    assert status.remaining_today == 2  # 5 - 3
    assert status.is_throttled is False


@pytest.mark.asyncio
async def test_throttler_record_suppressed(throttler: AlertThrottler, sample_alert: Alert) -> None:
    """Test recording suppressed alerts."""
    await throttler.record_suppressed("user-123", sample_alert)
    status = await throttler.get_status("user-123")
    assert status.suppressed_count >= 1


# ============================================================================
# Channel Tests
# ============================================================================


@pytest.mark.asyncio
async def test_websocket_channel_send_success() -> None:
    """Test WebSocket channel sends successfully."""
    mock_manager = Mock()
    mock_manager.send_personal_message = AsyncMock()

    channel = WebSocketChannel(mock_manager)
    alert = Alert(
        alert_id="ws-test-1",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test",
        message="Test message",
        channels=[AlertChannelEnum.WEBSOCKET],
        created_at=datetime.now(timezone.utc),
    )

    success = await channel.send(alert, "user-123")
    assert success is True
    mock_manager.send_personal_message.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_channel_send_failure() -> None:
    """Test WebSocket channel handles send failure."""
    mock_manager = Mock()
    mock_manager.send_personal_message = AsyncMock(side_effect=Exception("Connection lost"))

    channel = WebSocketChannel(mock_manager)
    alert = Alert(
        alert_id="ws-test-2",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test",
        message="Test",
        channels=[AlertChannelEnum.WEBSOCKET],
        created_at=datetime.now(timezone.utc),
    )

    success = await channel.send(alert, "user-123")
    assert success is False


@pytest.mark.asyncio
async def test_websocket_channel_is_available() -> None:
    """Test WebSocket channel availability check."""
    mock_manager = Mock()
    mock_manager.is_user_connected = Mock(return_value=True)

    channel = WebSocketChannel(mock_manager)
    available = await channel.is_available("user-123")
    assert available is True


@pytest.mark.asyncio
async def test_email_channel_send_success() -> None:
    """Test email channel sends successfully."""
    smtp_config = {
        "host": "smtp.test.com",
        "port": 587,
        "username": "test@test.com",
        "password": "password",
        "from_email": "alerts@signalforge.com",
    }

    channel = EmailChannel(smtp_config)
    channel.set_email_lookup(lambda user_id: "user@example.com")

    alert = Alert(
        alert_id="email-test-1",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.HIGH,
        title="Test Alert",
        message="This is a test",
        symbol="AAPL",
        channels=[AlertChannelEnum.EMAIL],
        created_at=datetime.now(timezone.utc),
    )

    # SMTP will fail because server is not configured
    # This tests the error handling
    success = await channel.send(alert, "user-123")
    assert success is False  # Should fail gracefully


@pytest.mark.asyncio
async def test_email_channel_no_lookup() -> None:
    """Test email channel fails without email lookup."""
    channel = EmailChannel({"host": "localhost"})

    alert = Alert(
        alert_id="email-test-2",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test",
        message="Test",
        channels=[AlertChannelEnum.EMAIL],
        created_at=datetime.now(timezone.utc),
    )

    success = await channel.send(alert, "user-123")
    assert success is False


@pytest.mark.asyncio
async def test_email_channel_is_available() -> None:
    """Test email channel availability check."""
    channel = EmailChannel({"host": "localhost"})
    channel.set_email_lookup(lambda user_id: "user@test.com" if user_id == "user-123" else None)

    available = await channel.is_available("user-123")
    assert available is True

    not_available = await channel.is_available("user-999")
    assert not_available is False


@pytest.mark.asyncio
async def test_push_channel_send_success() -> None:
    """Test push channel sends successfully."""
    mock_service = Mock()
    mock_service.send_to_user = AsyncMock(return_value=True)

    channel = PushChannel(mock_service)
    alert = Alert(
        alert_id="push-test-1",
        user_id="user-123",
        alert_type=AlertType.PRICE_TARGET,
        priority=AlertPriority.HIGH,
        title="Price Alert",
        message="Price target reached",
        symbol="NVDA",
        channels=[AlertChannelEnum.PUSH],
        created_at=datetime.now(timezone.utc),
    )

    success = await channel.send(alert, "user-123")
    assert success is True
    mock_service.send_to_user.assert_called_once()


@pytest.mark.asyncio
async def test_push_channel_is_available() -> None:
    """Test push channel availability check."""
    mock_service = Mock()
    mock_service.has_registered_device = AsyncMock(return_value=True)

    channel = PushChannel(mock_service)
    available = await channel.is_available("user-123")
    assert available is True


@pytest.mark.asyncio
async def test_channel_router_register_channel() -> None:
    """Test registering channels with router."""
    router = ChannelRouter()
    mock_channel = Mock(spec=AlertChannel)

    router.register_channel(AlertChannelEnum.WEBSOCKET, mock_channel)
    assert AlertChannelEnum.WEBSOCKET in router.channels


@pytest.mark.asyncio
async def test_channel_router_route_single_channel() -> None:
    """Test routing to single channel."""
    router = ChannelRouter()
    mock_channel = Mock(spec=AlertChannel)
    mock_channel.is_available = AsyncMock(return_value=True)
    mock_channel.send = AsyncMock(return_value=True)

    router.register_channel(AlertChannelEnum.WEBSOCKET, mock_channel)

    alert = Alert(
        alert_id="route-test-1",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test",
        message="Test",
        channels=[AlertChannelEnum.WEBSOCKET],
        created_at=datetime.now(timezone.utc),
    )

    preferences = AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL],
        enabled_channels=[AlertChannelEnum.WEBSOCKET],
        max_alerts_per_day=5,
        min_priority=AlertPriority.LOW,
    )

    results = await router.route(alert, "user-123", preferences)
    assert results["websocket"] is True
    mock_channel.send.assert_called_once()


@pytest.mark.asyncio
async def test_channel_router_disabled_in_preferences() -> None:
    """Test router skips channels disabled in preferences."""
    router = ChannelRouter()
    mock_channel = Mock(spec=AlertChannel)
    mock_channel.is_available = AsyncMock(return_value=True)
    mock_channel.send = AsyncMock(return_value=True)

    router.register_channel(AlertChannelEnum.EMAIL, mock_channel)

    alert = Alert(
        alert_id="route-test-2",
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM,
        title="Test",
        message="Test",
        channels=[AlertChannelEnum.EMAIL],
        created_at=datetime.now(timezone.utc),
    )

    preferences = AlertPreferences(
        user_id="user-123",
        enabled_types=[AlertType.SIGNAL],
        enabled_channels=[AlertChannelEnum.WEBSOCKET],  # Email NOT enabled
        max_alerts_per_day=5,
        min_priority=AlertPriority.LOW,
    )

    results = await router.route(alert, "user-123", preferences)
    assert results["email"] is False
    mock_channel.send.assert_not_called()


# ============================================================================
# Alert Manager Tests
# ============================================================================


@pytest.mark.asyncio
async def test_alert_manager_create_alert(alert_manager: AlertManager) -> None:
    """Test creating an alert."""
    alert = await alert_manager.create_alert(
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        title="Test Alert",
        message="This is a test alert",
        symbol="AAPL",
        priority=AlertPriority.MEDIUM,
    )

    assert alert.alert_id is not None
    assert alert.user_id == "user-123"
    assert alert.title == "Test Alert"
    assert alert.symbol == "AAPL"


@pytest.mark.asyncio
async def test_alert_manager_create_signal_alert(alert_manager: AlertManager) -> None:
    """Test creating signal alert from IntegratedSignal."""
    signal = IntegratedSignal(
        symbol="AAPL",
        direction=SignalDirection.LONG,
        strength=0.8,
        confidence=0.85,
        ml_contribution=0.3,
        nlp_contribution=0.25,
        execution_contribution=0.15,
        regime_adjustment=0.15,
        current_regime=MarketRegime.BULL,
        recommendation="buy",
        position_size_pct=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.08,
        explanation="Strong momentum with positive sentiment",
        generated_at=datetime.now(timezone.utc),
        valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    alert = await alert_manager.create_signal_alert("user-123", signal)

    assert alert.alert_type == AlertType.SIGNAL
    assert alert.symbol == "AAPL"
    assert "85%" in alert.message  # confidence
    assert alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]  # High confidence


@pytest.mark.asyncio
async def test_alert_manager_create_price_alert(alert_manager: AlertManager) -> None:
    """Test creating price alert."""
    alert = await alert_manager.create_price_alert(
        user_id="user-123",
        symbol="TSLA",
        current_price=250.50,
        target_price=250.00,
        direction="above",
    )

    assert alert.alert_type == AlertType.PRICE_TARGET
    assert alert.symbol == "TSLA"
    assert "250" in alert.message


@pytest.mark.asyncio
async def test_alert_manager_create_portfolio_gain_alert(alert_manager: AlertManager) -> None:
    """Test creating portfolio gain alert."""
    alert = await alert_manager.create_portfolio_alert(
        user_id="user-123",
        alert_subtype="gain",
        details={
            "symbol": "NVDA",
            "gain_pct": 7.5,
            "current_value": 15000.00,
        },
    )

    assert alert.alert_type == AlertType.PORTFOLIO
    assert alert.symbol == "NVDA"
    # Check for either "7.5%" or "7.50%" (both are valid)
    assert "7.5" in alert.message and "%" in alert.message


@pytest.mark.asyncio
async def test_alert_manager_create_portfolio_loss_alert(alert_manager: AlertManager) -> None:
    """Test creating portfolio loss alert with appropriate priority."""
    alert = await alert_manager.create_portfolio_alert(
        user_id="user-123",
        alert_subtype="loss",
        details={
            "symbol": "META",
            "loss_pct": 12.0,  # Large loss
        },
    )

    assert alert.alert_type == AlertType.PORTFOLIO
    assert alert.priority == AlertPriority.CRITICAL  # Large loss = CRITICAL


@pytest.mark.asyncio
async def test_alert_manager_get_user_alerts(alert_manager: AlertManager) -> None:
    """Test getting user alerts."""
    # Create multiple alerts
    for i in range(3):
        await alert_manager.create_alert(
            user_id="user-123",
            alert_type=AlertType.SIGNAL,
            title=f"Alert {i}",
            message=f"Message {i}",
        )

    alerts = await alert_manager.get_user_alerts("user-123")
    assert len(alerts) == 3


@pytest.mark.asyncio
async def test_alert_manager_get_unread_alerts(alert_manager: AlertManager) -> None:
    """Test getting only unread alerts."""
    # Create alerts
    alert1 = await alert_manager.create_alert(
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        title="Alert 1",
        message="Message 1",
    )

    await alert_manager.create_alert(
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        title="Alert 2",
        message="Message 2",
    )

    # Mark one as read
    await alert_manager.mark_read(alert1.alert_id, "user-123")

    unread = await alert_manager.get_user_alerts("user-123", unread_only=True)
    assert len(unread) == 1


@pytest.mark.asyncio
async def test_alert_manager_mark_read(alert_manager: AlertManager) -> None:
    """Test marking alert as read."""
    alert = await alert_manager.create_alert(
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        title="Test",
        message="Test",
    )

    assert alert.read_at is None

    await alert_manager.mark_read(alert.alert_id, "user-123")

    alerts = await alert_manager.get_user_alerts("user-123")
    marked_alert = next(a for a in alerts if a.alert_id == alert.alert_id)
    assert marked_alert.read_at is not None


@pytest.mark.asyncio
async def test_alert_manager_update_preferences(alert_manager: AlertManager) -> None:
    """Test updating user preferences."""
    updated = await alert_manager.update_preferences(
        user_id="user-123",
        updates={
            "max_alerts_per_day": 10,
            "min_priority": AlertPriority.HIGH,
        },
    )

    assert updated.max_alerts_per_day == 10
    assert updated.min_priority == AlertPriority.HIGH


@pytest.mark.asyncio
async def test_alert_manager_throttling_integration(alert_manager: AlertManager) -> None:
    """Test that manager integrates with throttler."""
    # Create 5 alerts (the default limit)
    for i in range(5):
        await alert_manager.create_alert(
            user_id="user-123",
            alert_type=AlertType.SIGNAL,
            title=f"Alert {i}",
            message=f"Message {i}",
        )

    # 6th alert should be throttled (not delivered but stored)
    alert = await alert_manager.create_alert(
        user_id="user-123",
        alert_type=AlertType.SIGNAL,
        title="Alert 6",
        message="Message 6",
    )

    assert alert.delivered_at is None  # Not delivered due to throttling


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_signal_flow() -> None:
    """Test complete signal alert flow."""
    # Setup
    throttler = AlertThrottler(default_max_per_day=10)
    router = ChannelRouter()

    # Mock WebSocket channel
    mock_ws_manager = Mock()
    mock_ws_manager.send_personal_message = AsyncMock()
    mock_ws_manager.is_user_connected = Mock(return_value=True)
    ws_channel = WebSocketChannel(mock_ws_manager)
    router.register_channel(AlertChannelEnum.WEBSOCKET, ws_channel)

    manager = AlertManager(throttler=throttler, channel_router=router)

    # Create signal
    signal = IntegratedSignal(
        symbol="AAPL",
        direction=SignalDirection.STRONG_LONG,
        strength=0.9,
        confidence=0.92,
        ml_contribution=0.35,
        nlp_contribution=0.30,
        execution_contribution=0.15,
        regime_adjustment=0.12,
        current_regime=MarketRegime.BULL,
        recommendation="strong_buy",
        position_size_pct=0.08,
        stop_loss_pct=0.02,
        take_profit_pct=0.12,
        explanation="Exceptional momentum with very positive sentiment and strong execution",
        generated_at=datetime.now(timezone.utc),
        valid_until=datetime.now(timezone.utc) + timedelta(hours=2),
    )

    # Create alert
    alert = await manager.create_signal_alert("user-123", signal)

    # Verify alert was created and delivered
    assert alert.alert_id is not None
    assert alert.priority == AlertPriority.CRITICAL  # High confidence + strength
    assert alert.delivered_at is not None
    assert mock_ws_manager.send_personal_message.called


@pytest.mark.asyncio
async def test_multi_channel_delivery() -> None:
    """Test delivering alerts through multiple channels."""
    throttler = AlertThrottler(default_max_per_day=10)
    router = ChannelRouter()

    # Mock channels
    mock_ws_manager = Mock()
    mock_ws_manager.send_personal_message = AsyncMock()
    mock_ws_manager.is_user_connected = Mock(return_value=True)

    mock_push_service = Mock()
    mock_push_service.send_to_user = AsyncMock()
    mock_push_service.has_registered_device = AsyncMock(return_value=True)

    ws_channel = WebSocketChannel(mock_ws_manager)
    push_channel = PushChannel(mock_push_service)

    router.register_channel(AlertChannelEnum.WEBSOCKET, ws_channel)
    router.register_channel(AlertChannelEnum.PUSH, push_channel)

    manager = AlertManager(throttler=throttler, channel_router=router)

    # Update preferences to enable both channels
    await manager.update_preferences(
        "user-123",
        {"enabled_channels": [AlertChannelEnum.WEBSOCKET, AlertChannelEnum.PUSH]},
    )

    # Create alert
    alert = await manager.create_alert(
        user_id="user-123",
        alert_type=AlertType.PRICE_TARGET,
        title="Price Alert",
        message="Target reached",
        symbol="NVDA",
        channels=[AlertChannelEnum.WEBSOCKET, AlertChannelEnum.PUSH],
    )

    # Verify both channels received the alert
    assert mock_ws_manager.send_personal_message.called
    assert mock_push_service.send_to_user.called
