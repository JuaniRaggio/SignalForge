"""WebSocket API routes for real-time updates."""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import WebSocketException

from signalforge.dashboard.config import DashboardConfigManager
from signalforge.dashboard.schemas import SubscribeMessage, WidgetUpdateMessage
from signalforge.dashboard.streaming import ConnectionManager, StreamingService
from signalforge.dashboard.widgets import WidgetFactory

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["websocket"])

# Global connection manager and streaming service
connection_manager = ConnectionManager()
streaming_service = StreamingService(connection_manager, WidgetFactory)


async def authenticate_websocket(token: str) -> dict[str, Any]:
    """Authenticate WebSocket connection via token.

    Args:
        token: Authentication token

    Returns:
        User information dictionary

    Raises:
        WebSocketException: If authentication fails
    """
    # TODO: Implement actual JWT token validation
    # For now, accept any token and extract user_id
    if not token:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Missing authentication token",
        )

    # Mock user validation - replace with actual JWT validation
    try:
        # In production, decode JWT and validate
        user_id = "test_user"  # Extract from token
        return {"user_id": user_id, "token": token}
    except Exception as e:
        logger.error("websocket.auth_failed", error=str(e))
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid authentication token",
        )


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token"),
) -> None:
    """WebSocket endpoint for real-time dashboard updates.

    This endpoint provides:
    - Real-time widget data updates
    - Price updates for subscribed symbols
    - Trading signal notifications
    - Alert notifications

    Message Types:
    - subscribe: Subscribe to channels (prices, signals, alerts)
    - unsubscribe: Unsubscribe from channels
    - widget_start: Start streaming updates for a widget
    - widget_stop: Stop streaming updates for a widget
    - ping: Keepalive ping

    Args:
        websocket: WebSocket connection
        token: JWT authentication token

    Example:
        Subscribe to price updates:
        ```json
        {
            "action": "subscribe",
            "channels": ["price:AAPL", "price:MSFT", "signals"]
        }
        ```

        Start widget updates:
        ```json
        {
            "action": "widget_start",
            "widget_id": "widget_123"
        }
        ```
    """
    # Authenticate connection
    try:
        user_info = await authenticate_websocket(token)
        user_id = user_info["user_id"]
    except WebSocketException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Accept and register connection
    await connection_manager.connect(websocket, user_id)

    logger.info(
        "websocket.connection_established",
        user_id=user_id,
    )

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "message_type": "connected",
                "payload": {
                    "user_id": user_id,
                    "message": "WebSocket connection established",
                },
            }
        )

        # Message handling loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)

                action = message.get("action")

                if action == "subscribe":
                    # Subscribe to channels
                    subscribe_msg = SubscribeMessage(**message)
                    for channel in subscribe_msg.channels:
                        await connection_manager.subscribe(user_id, channel)

                    await websocket.send_json(
                        {
                            "message_type": "subscribed",
                            "payload": {"channels": subscribe_msg.channels},
                        }
                    )

                    logger.info(
                        "websocket.subscribed",
                        user_id=user_id,
                        channels=subscribe_msg.channels,
                    )

                elif action == "unsubscribe":
                    # Unsubscribe from channels
                    subscribe_msg = SubscribeMessage(**message)
                    for channel in subscribe_msg.channels:
                        await connection_manager.unsubscribe(user_id, channel)

                    await websocket.send_json(
                        {
                            "message_type": "unsubscribed",
                            "payload": {"channels": subscribe_msg.channels},
                        }
                    )

                    logger.info(
                        "websocket.unsubscribed",
                        user_id=user_id,
                        channels=subscribe_msg.channels,
                    )

                elif action == "widget_start":
                    # Start widget updates
                    widget_msg = WidgetUpdateMessage(**message)
                    widget_id = widget_msg.widget_id

                    # Get widget configuration
                    config_manager = DashboardConfigManager()
                    layout = await config_manager.get_layout(user_id)

                    # Find widget config
                    widget_config = None
                    for widget in layout.widgets:
                        if widget.widget_id == widget_id:
                            widget_config = widget
                            break

                    if widget_config:
                        await streaming_service.start_widget_updates(
                            user_id, widget_id, widget_config
                        )

                        await websocket.send_json(
                            {
                                "message_type": "widget_started",
                                "payload": {"widget_id": widget_id},
                            }
                        )

                        logger.info(
                            "websocket.widget_started",
                            user_id=user_id,
                            widget_id=widget_id,
                        )
                    else:
                        await websocket.send_json(
                            {
                                "message_type": "error",
                                "payload": {
                                    "error": f"Widget {widget_id} not found"
                                },
                            }
                        )

                elif action == "widget_stop":
                    # Stop widget updates
                    widget_msg = WidgetUpdateMessage(**message)
                    widget_id = widget_msg.widget_id

                    await streaming_service.stop_widget_updates(user_id, widget_id)

                    await websocket.send_json(
                        {
                            "message_type": "widget_stopped",
                            "payload": {"widget_id": widget_id},
                        }
                    )

                    logger.info(
                        "websocket.widget_stopped",
                        user_id=user_id,
                        widget_id=widget_id,
                    )

                elif action == "ping":
                    # Keepalive ping
                    await websocket.send_json(
                        {
                            "message_type": "pong",
                            "payload": {},
                        }
                    )

                else:
                    # Unknown action
                    await websocket.send_json(
                        {
                            "message_type": "error",
                            "payload": {"error": f"Unknown action: {action}"},
                        }
                    )

                    logger.warning(
                        "websocket.unknown_action",
                        user_id=user_id,
                        action=action,
                    )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "message_type": "error",
                        "payload": {"error": "Invalid JSON"},
                    }
                )

            except Exception as e:
                logger.error(
                    "websocket.message_error",
                    user_id=user_id,
                    error=str(e),
                )
                await websocket.send_json(
                    {
                        "message_type": "error",
                        "payload": {"error": str(e)},
                    }
                )

    except WebSocketDisconnect:
        logger.info(
            "websocket.disconnected",
            user_id=user_id,
        )

    except Exception as e:
        logger.error(
            "websocket.error",
            user_id=user_id,
            error=str(e),
        )

    finally:
        # Clean up connection
        await connection_manager.disconnect(websocket, user_id)
        await streaming_service.stop_all_user_streams(user_id)

        logger.info(
            "websocket.connection_closed",
            user_id=user_id,
        )


@router.get("/ws/status")
async def websocket_status() -> dict[str, Any]:
    """Get WebSocket connection status.

    Returns:
        Status information including connected users count
    """
    connected_users = connection_manager.get_connected_users()

    return {
        "status": "operational",
        "connected_users": len(connected_users),
        "total_connections": sum(
            len(connection_manager.active_connections.get(user_id, []))
            for user_id in connected_users
        ),
    }
