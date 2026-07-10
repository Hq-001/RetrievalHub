"""健康检查与就绪探针路由。"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> JSONResponse:
    """存活探针 - 仅检查进程是否存活，不依赖外部资源。

    供 K8s livenessProbe / Docker HEALTHCHECK 使用。
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok"},
    )


@router.get("/readyz")
async def readyz() -> JSONResponse:
    """就绪探针 - 检查存储、嵌入等依赖是否连通。

    供 K8s readinessProbe 使用，未就绪返回 503 实现流量摘除。
    """
    # Phase 0 仅返回 ok，后续阶段接入真实依赖检查
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready"},
    )
