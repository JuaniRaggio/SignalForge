"""FastAPI application factory and main entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from signalforge.api.middleware.exception_handler import setup_exception_handlers
from signalforge.api.middleware.logging import LoggingMiddleware
from signalforge.api.middleware.metrics import MetricsMiddleware
from signalforge.api.routes import (
    api_keys,
    auth,
    billing,
    competition,
    dashboard,
    events,
    execution,
    explainability,
    health,
    leaderboard,
    market,
    ml,
    news,
    nlp,
    orderflow,
    paper_trading,
    recommendations,
    users,
    websocket,
)
from signalforge.core.config import get_settings
from signalforge.core.logging import configure_logging, get_logger
from signalforge.core.redis import close_redis

settings = get_settings()

# Configure structured logging
configure_logging(
    json_logs=settings.is_production,
    log_level="DEBUG" if settings.debug else "INFO",
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("application_startup", app_name=settings.app_name, env=settings.app_env)
    yield
    logger.info("application_shutdown")
    await close_redis()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Financial signal processing and market intelligence platform",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add metrics middleware (before logging to capture all requests)
    app.add_middleware(MetricsMiddleware)

    # Add structured logging middleware
    app.add_middleware(LoggingMiddleware)

    # Setup exception handlers for structured error responses
    setup_exception_handlers(app)

    app.include_router(health.router, tags=["Health"])
    app.include_router(
        auth.router,
        prefix=f"{settings.api_v1_prefix}/auth",
        tags=["Authentication"],
    )
    app.include_router(
        market.router,
        prefix=f"{settings.api_v1_prefix}/market",
        tags=["Market"],
    )
    app.include_router(
        news.router,
        prefix=f"{settings.api_v1_prefix}/news",
        tags=["News"],
    )
    app.include_router(
        users.router,
        prefix=f"{settings.api_v1_prefix}/users",
        tags=["Users"],
    )
    app.include_router(
        events.router,
        prefix=f"{settings.api_v1_prefix}",
        tags=["Events"],
    )
    app.include_router(
        orderflow.router,
        prefix=f"{settings.api_v1_prefix}",
        tags=["OrderFlow"],
    )
    app.include_router(
        ml.router,
        prefix=f"{settings.api_v1_prefix}/ml",
        tags=["Machine Learning"],
    )
    app.include_router(
        execution.router,
        prefix=f"{settings.api_v1_prefix}/execution",
        tags=["Execution Quality"],
    )
    app.include_router(
        nlp.router,
        prefix=f"{settings.api_v1_prefix}/nlp",
        tags=["NLP Signals"],
    )
    app.include_router(
        explainability.router,
        prefix=f"{settings.api_v1_prefix}/explainability",
        tags=["Explainability"],
    )
    app.include_router(
        recommendations.router,
        prefix=f"{settings.api_v1_prefix}/recommendations",
        tags=["Recommendations"],
    )
    app.include_router(
        billing.router,
        prefix=f"{settings.api_v1_prefix}/billing",
        tags=["Billing"],
    )
    app.include_router(
        api_keys.router,
        prefix=f"{settings.api_v1_prefix}",
        tags=["API Keys"],
    )
    app.include_router(
        dashboard.router,
        prefix=f"{settings.api_v1_prefix}",
        tags=["Dashboard"],
    )
    app.include_router(
        competition.router,
        tags=["Competitions"],
    )
    app.include_router(
        paper_trading.router,
        tags=["Paper Trading"],
    )
    app.include_router(
        leaderboard.router,
        tags=["Leaderboard"],
    )
    app.include_router(
        websocket.router,
        tags=["WebSocket"],
    )

    return app


app = create_app()
