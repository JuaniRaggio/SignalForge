"""Tests for WebSocket functionality."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket

from signalforge.dashboard.schemas import StreamMessage, WidgetConfig, WidgetSize, WidgetType
from signalforge.dashboard.streaming import ConnectionManager, StreamingService
from signalforge.dashboard.widgets import WidgetFactory


class TestConnectionManager:
    """Test WebSocket connection manager."""

    @pytest.mark.asyncio
    async def test_connect_websocket(self) -> None:
        """Test connecting a WebSocket."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws, "user_123")

        assert "user_123" in manager.active_connections
        assert mock_ws in manager.active_connections["user_123"]
        mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self) -> None:
        """Test disconnecting a WebSocket."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws, "user_123")
        await manager.disconnect(mock_ws, "user_123")

        assert "user_123" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_multiple_connections_same_user(self) -> None:
        """Test multiple connections for same user."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws1, "user_123")
        await manager.connect(mock_ws2, "user_123")

        assert len(manager.active_connections["user_123"]) == 2

    @pytest.mark.asyncio
    async def test_send_to_user(self) -> None:
        """Test sending message to user."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws, "user_123")

        message = StreamMessage(
            message_type="test",
            payload={"data": "test"},
            timestamp=datetime.utcnow(),
        )

        await manager.send_to_user("user_123", message)

        mock_ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_user(self) -> None:
        """Test sending message to nonexistent user does not raise."""
        manager = ConnectionManager()

        message = StreamMessage(
            message_type="test",
            payload={"data": "test"},
            timestamp=datetime.utcnow(),
        )

        # Should not raise
        await manager.send_to_user("nonexistent_user", message)

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self) -> None:
        """Test broadcasting to all users."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws1, "user_1")
        await manager.connect(mock_ws2, "user_2")

        message = StreamMessage(
            message_type="broadcast",
            payload={"data": "test"},
            timestamp=datetime.utcnow(),
        )

        await manager.broadcast(message)

        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self) -> None:
        """Test subscribing to a channel."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws, "user_123")
        await manager.subscribe("user_123", "price:AAPL")

        assert "price:AAPL" in manager.subscriptions["user_123"]

    @pytest.mark.asyncio
    async def test_unsubscribe_from_channel(self) -> None:
        """Test unsubscribing from a channel."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws, "user_123")
        await manager.subscribe("user_123", "price:AAPL")
        await manager.unsubscribe("user_123", "price:AAPL")

        assert "price:AAPL" not in manager.subscriptions["user_123"]

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self) -> None:
        """Test broadcasting to specific channel."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws1, "user_1")
        await manager.connect(mock_ws2, "user_2")

        await manager.subscribe("user_1", "signals")

        message = StreamMessage(
            message_type="signal",
            payload={"symbol": "AAPL"},
            timestamp=datetime.utcnow(),
        )

        await manager.broadcast(message, channel="signals")

        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_connected_users(self) -> None:
        """Test getting list of connected users."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws1, "user_1")
        await manager.connect(mock_ws2, "user_2")

        users = manager.get_connected_users()

        assert len(users) == 2
        assert "user_1" in users
        assert "user_2" in users


class TestStreamingService:
    """Test streaming service."""

    @pytest.mark.asyncio
    async def test_create_streaming_service(self) -> None:
        """Test creating streaming service."""
        manager = ConnectionManager()
        service = StreamingService(manager, WidgetFactory)

        assert service.connection_manager == manager
        assert service.widget_factory == WidgetFactory

    @pytest.mark.asyncio
    async def test_start_widget_updates(self) -> None:
        """Test starting widget updates."""
        manager = ConnectionManager()
        service = StreamingService(manager, WidgetFactory)

        widget_config = WidgetConfig(
            widget_id="widget_123",
            widget_type=WidgetType.PORTFOLIO_SUMMARY,
            size=WidgetSize.MEDIUM,
            position=(0, 0),
            refresh_interval=30,
        )

        await service.start_widget_updates("user_123", "widget_123", widget_config)

        # Check that task was created
        assert "user_123" in service._active_streams
        assert "widget_123" in service._active_streams["user_123"]

        # Clean up
        await service.stop_widget_updates("user_123", "widget_123")

    @pytest.mark.asyncio
    async def test_stop_widget_updates(self) -> None:
        """Test stopping widget updates."""
        manager = ConnectionManager()
        service = StreamingService(manager, WidgetFactory)

        widget_config = WidgetConfig(
            widget_id="widget_123",
            widget_type=WidgetType.WATCHLIST,
            size=WidgetSize.MEDIUM,
            position=(0, 0),
            refresh_interval=30,
        )

        await service.start_widget_updates("user_123", "widget_123", widget_config)
        await service.stop_widget_updates("user_123", "widget_123")

        assert "widget_123" not in service._active_streams.get("user_123", {})

    @pytest.mark.asyncio
    async def test_push_price_update(self) -> None:
        """Test pushing price update."""
        manager = ConnectionManager()
        service = StreamingService(manager, WidgetFactory)

        mock_ws = AsyncMock(spec=WebSocket)
        await manager.connect(mock_ws, "user_123")
        await manager.subscribe("user_123", "price:AAPL")

        await service.push_price_update("AAPL", 185.50, 2.50)

        mock_ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_signal_update(self) -> None:
        """Test pushing signal update."""
        manager = ConnectionManager()
        service = StreamingService(manager, WidgetFactory)

        mock_ws = AsyncMock(spec=WebSocket)
        await manager.connect(mock_ws, "user_123")
        await manager.subscribe("user_123", "signals")

        signal = {
            "signal_type": "buy",
            "strength": 0.85,
            "price": 185.50,
        }

        await service.push_signal_update("AAPL", signal)

        mock_ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_alert(self) -> None:
        """Test pushing alert to user."""
        manager = ConnectionManager()
        service = StreamingService(manager, WidgetFactory)

        mock_ws = AsyncMock(spec=WebSocket)
        await manager.connect(mock_ws, "user_123")

        alert = {
            "type": "price",
            "message": "AAPL reached target price",
            "severity": "info",
        }

        await service.push_alert("user_123", alert)

        mock_ws.send_json.assert_called_once()
