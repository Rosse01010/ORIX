"""
rate_limit.py
─────────────
Simple in-memory sliding-window rate limiter for the API.

Prevents CPU overload from excessive recognition requests.
Uses a per-IP counter with configurable requests-per-minute.

For production clusters, replace with Redis-backed rate limiting.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.
    Tracks request timestamps per client IP and rejects requests
    that exceed settings.api_rate_limit per minute.
    """

    def __init__(self, app, rate_limit: int | None = None):
        super().__init__(app)
        self.rate_limit = rate_limit or settings.api_rate_limit
        self.window_seconds = 60
        # { ip: [timestamp, timestamp, ...] }
        self._requests: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only rate-limit recognition endpoints (heavy compute)
        path = request.url.path
        if not path.startswith("/api/recognition"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - self.window_seconds

        # Prune old entries
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > window_start]

        if len(self._requests[client_ip]) >= self.rate_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Try again later.",
                    "limit": self.rate_limit,
                    "window_seconds": self.window_seconds,
                },
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
