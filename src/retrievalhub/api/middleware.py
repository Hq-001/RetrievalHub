"""REST 鉴权与限流中间件。

- API Key 鉴权（初期，预留用户体系扩展）
- 并发限流（令牌桶）
- 健康探针路径豁免鉴权
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)

# 鉴权豁免路径
EXEMPT_PATHS = {"/healthz", "/readyz", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 鉴权中间件。

    从 Authorization: Bearer <key> 或 X-API-Key 头提取密钥。
    健康探针路径豁免。
    """

    def __init__(self, app, api_key: str = "") -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # 鉴权未配置则放行
        if not self._api_key:
            return await call_next(request)

        path = request.url.path

        # 豁免路径
        if path in EXEMPT_PATHS or path.startswith("/docs"):
            return await call_next(request)

        # 提取 API Key
        auth_header = request.headers.get("Authorization", "")
        x_api_key = request.headers.get("X-API-Key", "")

        provided_key = ""
        if auth_header.startswith("Bearer "):
            provided_key = auth_header[7:]
        elif x_api_key:
            provided_key = x_api_key

        if provided_key != self._api_key:
            logger.warning("auth_failed", path=path, client=request.client.host if request.client else "unknown")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """令牌桶限流中间件。

    按 IP 限流，每秒最多 max_rps 个请求。
    """

    def __init__(self, app, max_rps: int = 20, burst: int = 50) -> None:
        super().__init__(app)
        self._max_rps = max_rps
        self._burst = burst
        self._buckets: dict[str, list[float]] = defaultdict(lambda: [float(burst), time.time()])

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # 健康探针不限流
        if path in EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        bucket = self._buckets[client_ip]
        tokens = bucket[0]
        last_time = bucket[1]
        now = time.time()

        # 补充令牌
        elapsed = now - last_time
        tokens = min(self._burst, tokens + elapsed * self._max_rps)

        if tokens < 1:
            logger.warning("rate_limited", client_ip=client_ip, path=path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "1"},
            )

        bucket[0] = tokens - 1
        bucket[1] = now

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件 - 记录请求方法、路径、耗时、状态码。"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.time()
        response = await call_next(request)
        elapsed_ms = int((time.time() - start) * 1000)

        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )

        # 注入耗时响应头
        response.headers["X-Response-Time-ms"] = str(elapsed_ms)
        return response
