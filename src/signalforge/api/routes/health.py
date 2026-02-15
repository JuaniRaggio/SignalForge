"""Health check endpoints."""

from fastapi import APIRouter, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from signalforge.core.database import check_database_connection, check_timescaledb
from signalforge.core.redis import check_redis_connection

router = APIRouter()


class HealthResponse(BaseModel):
    """Basic health check response."""

    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness check response with dependency status."""

    status: str
    database: bool
    redis: bool
    timescaledb: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Service not ready",
        }
    },
)
async def readiness_check() -> ReadinessResponse:
    """Readiness check that verifies all dependencies."""
    db_ok = await check_database_connection()
    redis_ok = await check_redis_connection()
    timescale_ok = await check_timescaledb()

    all_ok = db_ok and redis_ok and timescale_ok

    return ReadinessResponse(
        status="ready" if all_ok else "degraded",
        database=db_ok,
        redis=redis_ok,
        timescaledb=timescale_ok,
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics endpoint.

    Returns metrics in Prometheus text-based exposition format.
    This endpoint should be scraped by Prometheus at regular intervals.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
