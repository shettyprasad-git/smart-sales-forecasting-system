"""
In-Memory Sliding-Window Rate Limiter for Phase 7 Production Hardening.

Provides lightweight abuse protection on sensitive and high-cost endpoints
(authentication and heavy simulation/monitoring scans) without requiring external
caching infrastructure such as Redis.

NOTE: This in-memory limiter is designed for single-instance or portfolio
deployments. Distributed, multi-process, or clustered production architectures
should deploy infrastructure-level rate limiting (e.g. Cloudflare, AWS WAF,
Nginx limit_req, or a distributed Redis token-bucket).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from fastapi import HTTPException, Request, status


class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter.
    Tracks timestamp occurrences within a 60-second window per client IP.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_identifier: str) -> bool:
        now = time.time()
        window_start = now - 60.0

        with self._lock:
            # Filter timestamps to keep only those within the active 60-second window
            active_timestamps = [t for t in self.requests[client_identifier] if t > window_start]

            if len(active_timestamps) >= self.rpm:
                self.requests[client_identifier] = active_timestamps
                return False

            active_timestamps.append(now)
            self.requests[client_identifier] = active_timestamps
            return True

    def reset(self) -> None:
        """Clear all tracked request history (useful for automated testing)."""
        with self._lock:
            self.requests.clear()


# Pre-configured singletons for sensitive application areas
auth_rate_limiter = InMemoryRateLimiter(requests_per_minute=30)
heavy_intelligence_rate_limiter = InMemoryRateLimiter(requests_per_minute=60)


def get_client_ip(request: Request) -> str:
    """Extract client IP safely from forwarded headers or direct connection."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take the first untrusted client IP
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown_client"


def rate_limit_auth(request: Request) -> None:
    """Dependency enforcing 30 requests/minute on authentication endpoints."""
    client_ip = get_client_ip(request)
    if not auth_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication requests. Please try again in one minute.",
            headers={"Retry-After": "60"},
        )


def rate_limit_heavy_intelligence(request: Request) -> None:
    """Dependency enforcing 60 requests/minute on compute-heavy intelligence endpoints."""
    client_ip = get_client_ip(request)
    if not heavy_intelligence_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for analytical operations. Please retry shortly.",
            headers={"Retry-After": "60"},
        )
