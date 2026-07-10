"""Phase 7 测试 - 鉴权、限流、请求日志中间件。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from retrievalhub.api.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)


@pytest.fixture
def app_no_auth():
    """无鉴权的应用（api_key 为空）。"""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_rps=100, burst=200)
    app.add_middleware(APIKeyMiddleware, api_key="")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/data")
    async def get_data():
        return {"data": "test"}

    return app


@pytest.fixture
def app_with_auth():
    """有鉴权的应用。"""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_rps=100, burst=200)
    app.add_middleware(APIKeyMiddleware, api_key="secret-key-123")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ready"}

    @app.get("/v1/data")
    async def get_data():
        return {"data": "test"}

    return app


# ---- 鉴权测试 ----


class TestAPIKeyAuth:
    def test_healthz_exempt(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_readyz_exempt(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.get("/readyz")
        assert resp.status_code == 200

    def test_no_key_returns_401(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.get("/v1/data")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.get(
            "/v1/data",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_correct_bearer_key(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.get(
            "/v1/data",
            headers={"Authorization": "Bearer secret-key-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == "test"

    def test_correct_x_api_key(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.get(
            "/v1/data",
            headers={"X-API-Key": "secret-key-123"},
        )
        assert resp.status_code == 200

    def test_no_auth_when_key_empty(self, app_no_auth):
        """api_key 为空时所有请求放行。"""
        client = TestClient(app_no_auth)
        resp = client.get("/v1/data")
        assert resp.status_code == 200


# ---- 限流测试 ----


class TestRateLimit:
    def test_healthz_not_limited(self, app_no_auth):
        """健康探针不受限流影响。"""
        client = TestClient(app_no_auth)
        for _ in range(50):
            resp = client.get("/healthz")
            assert resp.status_code == 200

    def test_rate_limit_triggered(self):
        """超出 burst 限制后返回 429。"""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, max_rps=1, burst=3)
        app.add_middleware(APIKeyMiddleware, api_key="")

        @app.get("/v1/data")
        async def get_data():
            return {"data": "test"}

        client = TestClient(app)

        # 前 3 个请求通过（burst=3）
        for i in range(3):
            resp = client.get("/v1/data")
            assert resp.status_code == 200

        # 第 4 个应被限流
        resp = client.get("/v1/data")
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") is not None


# ---- 请求日志测试 ----


class TestRequestLogging:
    def test_response_time_header(self, app_no_auth):
        """响应应包含 X-Response-Time-ms 头。"""
        client = TestClient(app_no_auth)
        resp = client.get("/healthz")
        assert "x-response-time-ms" in resp.headers
        assert int(resp.headers["x-response-time-ms"]) >= 0

    def test_response_time_is_int(self, app_no_auth):
        client = TestClient(app_no_auth)
        resp = client.get("/healthz")
        # 应为整数字符串
        assert resp.headers["x-response-time-ms"].isdigit()
