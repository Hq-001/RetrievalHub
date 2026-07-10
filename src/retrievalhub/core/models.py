"""核心领域模型 - 知识库、文档、分块。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel as PydanticBaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---- 枚举 ----


class ContentType(str, Enum):
    """分块内容类型。"""

    MD = "md"
    JSON = "json"


class DocFormat(str, Enum):
    """文档格式。"""

    MD = "md"
    JSON = "json"
    JSONL = "jsonl"


class DocStatus(str, Enum):
    """文档解析状态。"""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


# ---- 实体 ----


class KnowledgeBase(PydanticBaseModel):
    """知识库实体。"""

    id: str
    name: str
    description: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    active_collection: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    model_config = {"from_attributes": True}


class Document(PydanticBaseModel):
    """文档实体。"""

    id: str
    kb_id: str
    filename: str
    file_format: DocFormat
    file_size: int = 0
    content_hash: str = ""
    status: DocStatus = DocStatus.PROCESSING
    started_at: datetime | None = None
    attempt_count: int = 0
    chunk_count: int = 0
    error_message: str | None = None
    format_version: str = "1"
    created_at: datetime = Field(default_factory=utcnow)

    model_config = {"from_attributes": True}


class Chunk(PydanticBaseModel):
    """分块实体 - 最小检索单元。"""

    id: str
    doc_id: str
    kb_id: str
    seq: int = 0
    text: str = ""
    content_type: ContentType = ContentType.MD
    section_path: str = ""  # MD: 标题路径，JSON: 空
    json_parent_id: str | None = None  # JSON: 父路径哈希，MD: None
    code_language: str | None = None  # 代码块语言标识，非代码块为 None
    char_offset: int = 0

    model_config = {"from_attributes": True}


# ---- DTO (检索) ----


class SearchHit(PydanticBaseModel):
    """单个检索结果。"""

    chunk_id: str
    text: str
    score: float
    content_type: ContentType
    section_path: str = ""
    json_parent_id: str | None = None
    code_language: str | None = None
    char_offset: int = 0
    doc_id: str
    doc_name: str = ""


class SearchRequest(PydanticBaseModel):
    """检索请求。"""

    query: str
    kb_id: str
    top_k: int = 10
    rerank: bool = True
    filters: dict | None = None
    timeout_ms: int = 5000


class SearchResponse(PydanticBaseModel):
    """检索响应 - 结构化 Chunk 列表，不含生成内容。"""

    hits: list[SearchHit]
    total: int = 0
    cached: bool = False
