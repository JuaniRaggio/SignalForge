"""Dashboard API routes."""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_user
from signalforge.api.dependencies.database import get_db
from signalforge.dashboard.config import DashboardConfigManager
from signalforge.dashboard.schemas import (
    DashboardLayoutResponse,
    WidgetConfigResponse,
    WidgetDataResponse,
    WidgetSize,
    WidgetType,
)
from signalforge.dashboard.widgets import WidgetFactory

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/layout", response_model=DashboardLayoutResponse)
async def get_dashboard_layout(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
    layout_name: str | None = Query(None, description="Layout name"),
) -> DashboardLayoutResponse:
    """Get user's dashboard layout.

    Args:
        layout_name: Optional layout name, defaults to user's default
        current_user: Current authenticated user
        session: Database session

    Returns:
        Dashboard layout configuration
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.get_layout",
        user_id=user_id,
        layout_name=layout_name,
    )

    config_manager = DashboardConfigManager(session=session)
    layout = await config_manager.get_layout(user_id, layout_name)

    return DashboardLayoutResponse(layout=layout, status="success")


@router.put("/layout", response_model=DashboardLayoutResponse)
async def save_dashboard_layout(
    _layout_request: dict[str, Any],
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DashboardLayoutResponse:
    """Save dashboard layout.

    Args:
        _layout_request: Layout data to save (TODO: implement parsing)
        current_user: Current authenticated user
        session: Database session

    Returns:
        Saved dashboard layout
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.save_layout",
        user_id=user_id,
    )

    # TODO: Parse _layout_request properly with Pydantic
    config_manager = DashboardConfigManager(session=session)

    # For now, get existing layout
    layout = await config_manager.get_layout(user_id)
    saved_layout = await config_manager.save_layout(layout)

    return DashboardLayoutResponse(layout=saved_layout, status="success")


@router.get("/widgets/{widget_id}/data", response_model=WidgetDataResponse)
async def get_widget_data(
    widget_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> WidgetDataResponse:
    """Get data for a specific widget.

    Args:
        widget_id: Widget identifier
        current_user: Current authenticated user
        session: Database session

    Returns:
        Widget data

    Raises:
        HTTPException: If widget not found
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.get_widget_data",
        user_id=user_id,
        widget_id=widget_id,
    )

    # Get user's layout to find the widget
    config_manager = DashboardConfigManager(session=session)
    layout = await config_manager.get_layout(user_id)

    # Find widget config
    widget_config = None
    for widget in layout.widgets:
        if widget.widget_id == widget_id:
            widget_config = widget
            break

    if widget_config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Widget {widget_id} not found",
        )

    # Get widget data
    widget_instance = WidgetFactory.create(widget_config.widget_type, session=session)
    data = await widget_instance.get_data(user_id, widget_config.settings)

    # Create response
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    from signalforge.dashboard.schemas import WidgetData

    widget_data = WidgetData(
        widget_id=widget_id,
        widget_type=widget_config.widget_type,
        data=data,
        last_updated=now,
        next_update=now + timedelta(seconds=widget_config.refresh_interval),
    )

    return WidgetDataResponse(widget_data=widget_data, status="success")


@router.post("/widgets", response_model=WidgetConfigResponse)
async def add_widget(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
    widget_type: WidgetType = Query(..., description="Widget type"),
    row: int = Query(..., description="Row position", ge=0),
    col: int = Query(..., description="Column position", ge=0),
    size: WidgetSize = Query(WidgetSize.MEDIUM, description="Widget size"),
) -> WidgetConfigResponse:
    """Add widget to dashboard.

    Args:
        widget_type: Type of widget to add
        row: Row position
        col: Column position
        size: Widget size
        current_user: Current authenticated user
        session: Database session

    Returns:
        Created widget configuration
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.add_widget",
        user_id=user_id,
        widget_type=widget_type.value,
        position=(row, col),
    )

    config_manager = DashboardConfigManager(session=session)
    widget_config = await config_manager.add_widget(
        user_id=user_id,
        widget_type=widget_type,
        position=(row, col),
        size=size,
    )

    return WidgetConfigResponse(widget_config=widget_config, status="success")


@router.delete("/widgets/{widget_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_widget(
    widget_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove widget from dashboard.

    Args:
        widget_id: Widget identifier to remove
        current_user: Current authenticated user
        session: Database session
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.remove_widget",
        user_id=user_id,
        widget_id=widget_id,
    )

    config_manager = DashboardConfigManager(session=session)
    await config_manager.remove_widget(user_id, widget_id)


@router.put("/widgets/{widget_id}/settings", response_model=WidgetConfigResponse)
async def update_widget_settings(
    widget_id: str,
    settings: dict[str, Any],
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> WidgetConfigResponse:
    """Update widget settings.

    Args:
        widget_id: Widget identifier
        settings: New settings to apply
        current_user: Current authenticated user
        session: Database session

    Returns:
        Updated widget configuration
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.update_widget_settings",
        user_id=user_id,
        widget_id=widget_id,
    )

    config_manager = DashboardConfigManager(session=session)
    widget_config = await config_manager.update_widget_settings(
        user_id, widget_id, settings
    )

    return WidgetConfigResponse(widget_config=widget_config, status="success")


@router.get("/widgets/available", response_model=list[WidgetType])
async def get_available_widgets(
    _current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> list[WidgetType]:
    """Get available widget types.

    Args:
        _current_user: Current authenticated user (unused, but required for auth)

    Returns:
        List of available widget types
    """
    logger.info("api.dashboard.get_available_widgets")

    return WidgetFactory.get_available_widgets()


@router.get("/layouts", response_model=list[dict[str, Any]])
async def list_layouts(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> list[dict[str, Any]]:
    """List all dashboard layouts for user.

    Args:
        current_user: Current authenticated user
        session: Database session

    Returns:
        List of layout summaries
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.list_layouts",
        user_id=user_id,
    )

    config_manager = DashboardConfigManager(session=session)
    layouts = await config_manager.list_layouts(user_id)

    # Return simplified layout info
    return [
        {
            "layout_name": layout.layout_name,
            "is_default": layout.is_default,
            "widget_count": len(layout.widgets),
            "updated_at": layout.updated_at.isoformat(),
        }
        for layout in layouts
    ]


@router.put("/widgets/{widget_id}/position", response_model=WidgetConfigResponse)
async def update_widget_position(
    widget_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
    row: int = Query(..., description="New row position", ge=0),
    col: int = Query(..., description="New column position", ge=0),
) -> WidgetConfigResponse:
    """Update widget position.

    Args:
        widget_id: Widget identifier
        row: New row position
        col: New column position
        current_user: Current authenticated user
        session: Database session

    Returns:
        Updated widget configuration
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.update_widget_position",
        user_id=user_id,
        widget_id=widget_id,
        position=(row, col),
    )

    config_manager = DashboardConfigManager(session=session)
    widget_config = await config_manager.update_widget_position(
        user_id, widget_id, (row, col)
    )

    return WidgetConfigResponse(widget_config=widget_config, status="success")


@router.put("/widgets/{widget_id}/size", response_model=WidgetConfigResponse)
async def update_widget_size(
    widget_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db)],
    size: WidgetSize = Query(..., description="New widget size"),
) -> WidgetConfigResponse:
    """Update widget size.

    Args:
        widget_id: Widget identifier
        size: New widget size
        current_user: Current authenticated user
        session: Database session

    Returns:
        Updated widget configuration
    """
    user_id = current_user.get("user_id", "anonymous")

    logger.info(
        "api.dashboard.update_widget_size",
        user_id=user_id,
        widget_id=widget_id,
        size=size.value,
    )

    config_manager = DashboardConfigManager(session=session)
    widget_config = await config_manager.update_widget_size(user_id, widget_id, size)

    return WidgetConfigResponse(widget_config=widget_config, status="success")
