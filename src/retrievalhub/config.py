"""配置加载模块 - 所有配置通过 .env 注入，零硬编码。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MDChunkConfig(BaseModel):
    """Markdown 分块参数（独立配置）。"""

    chunk_size: int = Field(default=800, description="MD 块大小上限（字符）")
    chunk_overlap: int = Field(default=100, description="MD 块重叠量（字符）")
    max_section_depth: int = Field(default=3, description="嵌套标题继承深度上限")


class JSONChunkConfig(BaseModel):
    """JSON 分块参数（独立配置）。"""

    chunk_size: int = Field(default=600, description="JSON 块大小上限（字符）")
    chunk_overlap: int = Field(default=60, description="JSON 块重叠量（字符）")
    long_value_threshold: int = Field(
        default=2000, description="超长 Value 阈值（tokens）"
    )


class Settings(BaseSettings):
    """全局配置，从 .env 加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 服务相关 ----
    rest_host: str = Field(default="0.0.0.0")
    rest_port: int = Field(default=8000)
    mcp_transport: str = Field(default="stdio")  # stdio | sse
    mcp_sse_port: int = Field(default=8001)
    log_level: str = Field(default="INFO")

    # ---- 上传限制 ----
    max_upload_size_mb: int = Field(default=10)
    max_concurrent_uploads: int = Field(default=5)

    # ---- 嵌入模型 ----
    embedding_endpoint: str = Field(default="")
    embedding_api_key: str = Field(default="")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dim: int = Field(default=1536)
    embedding_batch_size: int = Field(default=64)

    # ---- 重排序模型 ----
    reranker_enabled: bool = Field(default=True)
    reranker_endpoint: str = Field(default="")
    reranker_api_key: str = Field(default="")
    reranker_model: str = Field(default="")
    reranker_top_k: int = Field(default=20)

    # ---- 存储 (LanceDB) ----
    lancedb_uri: str = Field(default="./data/lancedb")
    lancedb_distance_metric: str = Field(default="cosine")

    # ---- 元数据库 (SQLite) ----
    sqlite_path: str = Field(default="./data/metadata.db")

    # ---- 分块参数 ----
    md_chunk_size: int = Field(default=800, alias="MD_CHUNK_SIZE")
    md_chunk_overlap: int = Field(default=100, alias="MD_CHUNK_OVERLAP")
    md_max_section_depth: int = Field(default=3, alias="MD_MAX_SECTION_DEPTH")
    json_chunk_size: int = Field(default=600, alias="JSON_CHUNK_SIZE")
    json_chunk_overlap: int = Field(default=60, alias="JSON_CHUNK_OVERLAP")
    json_long_value_threshold: int = Field(
        default=2000, alias="JSON_LONG_VALUE_THRESHOLD"
    )

    # ---- 检索参数 ----
    vector_recall_top_n: int = Field(default=50)
    bm25_recall_top_n: int = Field(default=50)
    rerank_top_k: int = Field(default=10)

    # ---- 动态加权 ----
    md_vector_weight: float = Field(default=1.1)
    json_bm25_weight: float = Field(default=1.2)

    # ---- 异步入库 ----
    ingest_timeout_sec: int = Field(default=30)

    # ---- 热点查询 LRU 缓存 ----
    query_cache_maxsize: int = Field(default=512)
    query_cache_ttl_sec: int = Field(default=300)

    # ---- 存储加密 ----
    enable_storage_encryption: bool = Field(default=False)
    storage_encryption_key: str = Field(default="")

    # ---- MCP 超时与重试 ----
    mcp_search_timeout_ms: int = Field(default=5000)

    # ---- 鉴权 ----
    api_key: str = Field(default="")

    @property
    def md_chunk_config(self) -> MDChunkConfig:
        return MDChunkConfig(
            chunk_size=self.md_chunk_size,
            chunk_overlap=self.md_chunk_overlap,
            max_section_depth=self.md_max_section_depth,
        )

    @property
    def json_chunk_config(self) -> JSONChunkConfig:
        return JSONChunkConfig(
            chunk_size=self.json_chunk_size,
            chunk_overlap=self.json_chunk_overlap,
            long_value_threshold=self.json_long_value_threshold,
        )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def data_dir(self) -> Path:
        p = Path("./data")
        p.mkdir(parents=True, exist_ok=True)
        return p


# 全局单例
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """重置全局配置（测试用）。"""
    global _settings
    _settings = None
