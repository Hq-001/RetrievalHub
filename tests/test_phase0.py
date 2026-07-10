"""Phase 0 验收测试 - 配置加载、模型校验、健康探针。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from retrievalhub.app import create_app
from retrievalhub.config import Settings, get_settings, reset_settings
from retrievalhub.core.models import (
    Chunk,
    ContentType,
    DocFormat,
    DocStatus,
    Document,
    KnowledgeBase,
)


# ---- 配置测试 ----


class TestSettings:
    def test_default_settings_load(self):
        """默认配置可加载。"""
        reset_settings()
        s = get_settings()
        assert s.rest_port == 8000
        assert s.mcp_transport == "stdio"
        assert s.md_chunk_size == 800
        assert s.json_chunk_size == 600
        assert s.md_vector_weight == 1.1
        assert s.json_bm25_weight == 1.2

    def test_md_chunk_config(self):
        s = Settings()
        cfg = s.md_chunk_config
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 100
        assert cfg.max_section_depth == 3

    def test_json_chunk_config(self):
        s = Settings()
        cfg = s.json_chunk_config
        assert cfg.chunk_size == 600
        assert cfg.chunk_overlap == 60
        assert cfg.long_value_threshold == 2000

    def test_max_upload_size_bytes(self):
        s = Settings()
        assert s.max_upload_size_bytes == 10 * 1024 * 1024


# ---- 模型测试 ----


class TestModels:
    def test_knowledge_base_defaults(self):
        kb = KnowledgeBase(id="kb1", name="测试知识库")
        assert kb.id == "kb1"
        assert kb.name == "测试知识库"
        assert kb.created_at is not None

    def test_document_defaults(self):
        doc = Document(
            id="doc1",
            kb_id="kb1",
            filename="test.md",
            file_format=DocFormat.MD,
        )
        assert doc.status == DocStatus.PROCESSING
        assert doc.attempt_count == 0
        assert doc.format_version == "1"

    def test_chunk_defaults(self):
        chunk = Chunk(id="c1", doc_id="doc1", kb_id="kb1")
        assert chunk.content_type == ContentType.MD
        assert chunk.section_path == ""
        assert chunk.json_parent_id is None
        assert chunk.code_language is None

    def test_content_type_enum(self):
        assert ContentType.MD == "md"
        assert ContentType.JSON == "json"

    def test_doc_status_enum(self):
        assert DocStatus.PROCESSING == "processing"
        assert DocStatus.READY == "ready"
        assert DocStatus.FAILED == "failed"


# ---- 健康探针测试 ----


class TestHealthEndpoints:
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readyz(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_openapi_docs(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
