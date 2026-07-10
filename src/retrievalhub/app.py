"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from retrievalhub.api.health import router as health_router
from retrievalhub.config import get_settings
from retrievalhub.utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info("retrievalhub_starting", transport=settings.mcp_transport)

    yield

    logger.info("retrievalhub_stopping")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    app = FastAPI(
        title="RetrievalHub",
        description="个人智能知识库检索中间件 - RAG-Ready 检索基础设施",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health_router)

    return app


app = create_app()
