"""Metrics middleware for automatic request tracking."""

from time import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from signalforge.core.metrics import active_requests, request_latency_seconds, request_total


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically collect metrics for all HTTP requests.

    This middleware tracks:
    - Request latency (histogram)
    - Total requests (counter with status code)
    - Active requests (gauge)

    Excludes metrics and health endpoints to avoid metric pollution.
    """

    EXCLUDED_PATHS = {"/metrics", "/health", "/health/ready"}

    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            Response: The HTTP response
        """
        path = request.url.path

        # Skip metrics collection for excluded endpoints
        if path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Extract endpoint and method for labeling
        endpoint = self._get_endpoint(request)
        method = request.method

        # Track active requests
        active_requests.inc()
        start_time = time()

        try:
            # Process the request
            response: Response = await call_next(request)
            status_code = response.status_code

            # Record metrics
            duration = time() - start_time
            request_latency_seconds.labels(endpoint=endpoint, method=method).observe(duration)
            request_total.labels(
                endpoint=endpoint, method=method, status=str(status_code)
            ).inc()

            return response

        except Exception as exc:
            # Track errors with 500 status
            duration = time() - start_time
            request_latency_seconds.labels(endpoint=endpoint, method=method).observe(duration)
            request_total.labels(endpoint=endpoint, method=method, status="500").inc()
            raise exc

        finally:
            # Always decrement active requests
            active_requests.dec()

    def _get_endpoint(self, request: Request) -> str:
        """Extract the endpoint path from the request.

        Attempts to get the route pattern if available, otherwise uses the path.
        This groups similar endpoints together (e.g., /users/{id} instead of /users/123).

        Args:
            request: The incoming HTTP request

        Returns:
            str: The endpoint path or route pattern
        """
        # Try to get the route pattern from matched route
        if request.scope.get("route"):
            return request.scope["route"].path

        # Fallback to raw path
        return request.url.path
