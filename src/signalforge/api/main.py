"""FastAPI application factory and main entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from signalforge.api.routes import auth, health, market, news
from signalforge.core.config import get_settings
from signalforge.core.redis import close_redis

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    yield
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
