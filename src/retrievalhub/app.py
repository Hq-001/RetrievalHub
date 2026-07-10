"""FastAPI 应用入口 - 整合 REST + MCP 双通道。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from retrievalhub.api.documents import router as documents_router, set_dependencies
from retrievalhub.api.health import router as health_router
from retrievalhub.api.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from retrievalhub.config import get_settings
from retrievalhub.embedders.embedder import MockEmbedder
from retrievalhub.ingest.enqueue import InProcessEnqueueBackend
from retrievalhub.ingest.crash_recovery import CrashRecovery
from retrievalhub.ingest.pipeline import IngestPipeline
from retrievalhub.mcp_server.tool import McpSearchTool
from retrievalhub.retrieval.service import SearchService
from retrievalhub.storage.lancedb_store import LanceDBStorage
from retrievalhub.storage.metadata_store import MetadataStore
from retrievalhub.utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 初始化所有组件。"""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info("retrievalhub_starting", transport=settings.mcp_transport)

    # 初始化存储
    storage = LanceDBStorage(
        uri=settings.lancedb_uri,
        distance_metric=settings.lancedb_distance_metric,
    )
    metadata = MetadataStore(db_path=settings.sqlite_path)

    # 崩溃恢复：启动时扫描超时 processing 文档
    recovery = CrashRecovery(metadata, timeout_sec=settings.ingest_timeout_sec)
    recovered = recovery.recover_stale_documents()
    if recovered > 0:
        logger.info("crash_recovery_on_startup", recovered=recovered)

    # 初始化嵌入器
    embedder = MockEmbedder(dim=settings.embedding_dim)

    # 初始化入库流水线
    pipeline = IngestPipeline(
        storage=storage,
        metadata=metadata,
        embedder=embedder,
        md_chunk_size=settings.md_chunk_size,
        md_chunk_overlap=settings.md_chunk_overlap,
        max_section_depth=settings.md_max_section_depth,
        json_chunk_size=settings.json_chunk_size,
        json_chunk_overlap=settings.json_chunk_overlap,
        json_long_value_threshold=settings.json_long_value_threshold,
    )

    # 初始化异步后端
    enqueue = InProcessEnqueueBackend(timeout_sec=settings.ingest_timeout_sec)

    # 原文存储目录
    original_dir = Path("./data/originals")
    original_dir.mkdir(parents=True, exist_ok=True)

    # 注入 REST 依赖
    set_dependencies(storage, metadata, embedder, pipeline, enqueue, original_dir)

    # 初始化检索服务
    search_service = SearchService(
        storage=storage,
        metadata=metadata,
        embedder=embedder,
        md_vector_weight=settings.md_vector_weight,
        json_bm25_weight=settings.json_bm25_weight,
        vector_top_n=settings.vector_recall_top_n,
        bm25_top_n=settings.bm25_recall_top_n,
        rerank_top_k=settings.rerank_top_k,
    )

    # 初始化 MCP 工具
    mcp_tool = McpSearchTool(search_service)

    # 挂载到 app.state
    app.state.storage = storage
    app.state.metadata = metadata
    app.state.embedder = embedder
    app.state.pipeline = pipeline
    app.state.enqueue = enqueue
    app.state.search_service = search_service
    app.state.mcp_tool = mcp_tool

    yield

    metadata.close()
    logger.info("retrievalhub_stopping")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    settings = get_settings()

    app = FastAPI(
        title="RetrievalHub",
        description="个人智能知识库检索中间件 - RAG-Ready 检索基础设施",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 中间件（按注册顺序逆序执行：最后注册的最先执行）
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_rps=20, burst=50)
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)

    app.include_router(health_router)
    app.include_router(documents_router)

    return app


app = create_app()
