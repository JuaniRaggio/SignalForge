"""Real-time streaming service for dashboard updates."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.dashboard.schemas import StreamMessage, WidgetConfig
from signalforge.dashboard.widgets import WidgetFactory

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.subscriptions: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """Accept and register a WebSocket connection.

        Args:
            websocket: WebSocket connection to accept
            user_id: User identifier
        """
        await websocket.accept()

        async with self._lock:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
                self.subscriptions[user_id] = set()

            self.active_connections[user_id].append(websocket)

        logger.info(
            "websocket.connected",
            user_id=user_id,
            total_connections=len(self.active_connections[user_id]),
        )

    async def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
            user_id: User identifier
        """
        async with self._lock:
            if user_id in self.active_connections:
                if websocket in self.active_connections[user_id]:
                    self.active_connections[user_id].remove(websocket)

                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                    if user_id in self.subscriptions:
                        del self.subscriptions[user_id]

        logger.info(
            "websocket.disconnected",
            user_id=user_id,
            remaining_connections=len(
                self.active_connections.get(user_id, [])
            ),
        )

    async def send_to_user(self, user_id: str, message: StreamMessage) -> None:
        """Send message to all user's connections.

        Args:
            user_id: User identifier
            message: Message to send
        """
        if user_id not in self.active_connections:
            return

        connections = self.active_connections[user_id].copy()
        disconnected = []

        for connection in connections:
            try:
                await connection.send_json(message.model_dump(mode="json"))
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(
                    "websocket.send_error",
                    user_id=user_id,
                    error=str(e),
                )
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            await self.disconnect(connection, user_id)

    async def broadcast(
        self, message: StreamMessage, channel: str | None = None
    ) -> None:
        """Broadcast message to all or specific channel subscribers.

        Args:
            message: Message to broadcast
            channel: Optional channel name to filter subscribers
        """
        for user_id in list(self.active_connections.keys()):
            # If channel specified, only send to subscribed users
            if channel is not None and (
                user_id not in self.subscriptions
                or channel not in self.subscriptions[user_id]
            ):
                continue

            await self.send_to_user(user_id, message)

    async def subscribe(self, user_id: str, channel: str) -> None:
        """Subscribe user to a channel.

        Args:
            user_id: User identifier
            channel: Channel name to subscribe to
        """
        async with self._lock:
            if user_id not in self.subscriptions:
                self.subscriptions[user_id] = set()

            self.subscriptions[user_id].add(channel)

        logger.info(
            "websocket.subscribe",
            user_id=user_id,
            channel=channel,
            total_channels=len(self.subscriptions[user_id]),
        )

    async def unsubscribe(self, user_id: str, channel: str) -> None:
        """Unsubscribe user from a channel.

        Args:
            user_id: User identifier
            channel: Channel name to unsubscribe from
        """
        async with self._lock:
            if user_id in self.subscriptions:
                self.subscriptions[user_id].discard(channel)

        logger.info(
            "websocket.unsubscribe",
            user_id=user_id,
            channel=channel,
        )

    def get_connected_users(self) -> list[str]:
        """Get list of connected user IDs.

        Returns:
            List of user IDs with active connections
        """
        return list(self.active_connections.keys())

    def get_user_channels(self, user_id: str) -> set[str]:
        """Get channels subscribed by user.

        Args:
            user_id: User identifier

        Returns:
            Set of channel names
        """
        return self.subscriptions.get(user_id, set()).copy()


class StreamingService:
    """Service for real-time data streaming to dashboard widgets."""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        widget_factory: type[WidgetFactory] | None = None,
    ) -> None:
        """Initialize streaming service.

        Args:
            connection_manager: Connection manager instance
            widget_factory: Widget factory class (defaults to WidgetFactory)
        """
        self.connection_manager = connection_manager
        self.widget_factory = widget_factory or WidgetFactory
        self._active_streams: dict[str, dict[str, asyncio.Task[None]]] = {}
        self._lock = asyncio.Lock()

    async def start_widget_updates(
        self,
        user_id: str,
        widget_id: str,
        widget_config: WidgetConfig,
        session: AsyncSession | None = None,
    ) -> None:
        """Start streaming updates for a widget.

        Args:
            user_id: User identifier
            widget_id: Widget identifier
            widget_config: Widget configuration
            session: Optional database session
        """
        async with self._lock:
            if user_id not in self._active_streams:
                self._active_streams[user_id] = {}

            # Stop existing stream if present
            if widget_id in self._active_streams[user_id]:
                self._active_streams[user_id][widget_id].cancel()

            # Create new streaming task
            task = asyncio.create_task(
                self._widget_update_loop(
                    user_id, widget_id, widget_config, session
                )
            )
            self._active_streams[user_id][widget_id] = task

        logger.info(
            "streaming.widget_started",
            user_id=user_id,
            widget_id=widget_id,
            widget_type=widget_config.widget_type.value,
            refresh_interval=widget_config.refresh_interval,
        )

    async def stop_widget_updates(self, user_id: str, widget_id: str) -> None:
        """Stop streaming updates for a widget.

        Args:
            user_id: User identifier
            widget_id: Widget identifier
        """
        async with self._lock:
            if (
                user_id in self._active_streams
                and widget_id in self._active_streams[user_id]
            ):
                self._active_streams[user_id][widget_id].cancel()
                del self._active_streams[user_id][widget_id]

                if not self._active_streams[user_id]:
                    del self._active_streams[user_id]

        logger.info(
            "streaming.widget_stopped",
            user_id=user_id,
            widget_id=widget_id,
        )

    async def _widget_update_loop(
        self,
        user_id: str,
        widget_id: str,
        widget_config: WidgetConfig,
        session: AsyncSession | None = None,
    ) -> None:
        """Internal loop for widget updates.

        Args:
            user_id: User identifier
            widget_id: Widget identifier
            widget_config: Widget configuration
            session: Optional database session
        """
        try:
            widget = self.widget_factory.create(
                widget_config.widget_type, session=session
            )

            while True:
                try:
                    # Get widget data
                    data = await widget.get_data(user_id, widget_config.settings)

                    # Create stream message
                    now = datetime.utcnow()
                    message = StreamMessage(
                        message_type="widget_update",
                        payload={
                            "widget_id": widget_id,
                            "widget_type": widget_config.widget_type.value,
                            "data": data,
                            "last_updated": now.isoformat(),
                            "next_update": (
                                now + timedelta(seconds=widget_config.refresh_interval)
                            ).isoformat(),
                        },
                        timestamp=now,
                    )

                    # Send to user
                    await self.connection_manager.send_to_user(user_id, message)

                    logger.debug(
                        "streaming.widget_updated",
                        user_id=user_id,
                        widget_id=widget_id,
                        widget_type=widget_config.widget_type.value,
                    )

                except Exception as e:
                    logger.error(
                        "streaming.widget_update_error",
                        user_id=user_id,
                        widget_id=widget_id,
                        error=str(e),
                    )

                # Wait for next update
                await asyncio.sleep(widget_config.refresh_interval)

        except asyncio.CancelledError:
            logger.debug(
                "streaming.widget_loop_cancelled",
                user_id=user_id,
                widget_id=widget_id,
            )
            raise

    async def push_price_update(
        self, symbol: str, price: float, change: float
    ) -> None:
        """Push price update to subscribers.

        Args:
            symbol: Stock symbol
            price: Current price
            change: Price change
        """
        message = StreamMessage(
            message_type="price",
            payload={
                "symbol": symbol,
                "price": price,
                "change": change,
                "change_pct": (change / (price - change)) * 100 if price != change else 0.0,
            },
            timestamp=datetime.utcnow(),
        )

        # Broadcast to price channel
        await self.connection_manager.broadcast(message, channel=f"price:{symbol}")

        logger.debug(
            "streaming.price_update",
            symbol=symbol,
            price=price,
            change=change,
        )

    async def push_signal_update(self, symbol: str, signal: dict[str, Any]) -> None:
        """Push new signal to relevant users.

        Args:
            symbol: Stock symbol
            signal: Signal data
        """
        message = StreamMessage(
            message_type="signal",
            payload={
                "symbol": symbol,
                **signal,
            },
            timestamp=datetime.utcnow(),
        )

        # Broadcast to signals channel
        await self.connection_manager.broadcast(message, channel="signals")

        logger.info(
            "streaming.signal_update",
            symbol=symbol,
            signal_type=signal.get("signal_type"),
        )

    async def push_alert(self, user_id: str, alert: dict[str, Any]) -> None:
        """Push alert to specific user.

        Args:
            user_id: User identifier
            alert: Alert data
        """
        message = StreamMessage(
            message_type="alert",
            payload=alert,
            timestamp=datetime.utcnow(),
        )

        await self.connection_manager.send_to_user(user_id, message)

        logger.info(
            "streaming.alert_pushed",
            user_id=user_id,
            alert_type=alert.get("type"),
        )

    async def stop_all_user_streams(self, user_id: str) -> None:
        """Stop all streaming updates for a user.

        Args:
            user_id: User identifier
        """
        async with self._lock:
            if user_id in self._active_streams:
                for task in self._active_streams[user_id].values():
                    task.cancel()
                del self._active_streams[user_id]

        logger.info(
            "streaming.all_stopped",
            user_id=user_id,
        )

    def get_active_widgets(self, user_id: str) -> list[str]:
        """Get list of active widget IDs for a user.

        Args:
            user_id: User identifier

        Returns:
            List of widget IDs
        """
        return list(self._active_streams.get(user_id, {}).keys())
