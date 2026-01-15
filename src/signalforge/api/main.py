"""FastAPI application factory and main entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from signalforge.api.middleware.exception_handler import setup_exception_handlers
from signalforge.api.middleware.logging import LoggingMiddleware
from signalforge.api.routes import auth, health, market, news
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

    return app


app = create_app()
