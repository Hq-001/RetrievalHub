"""Phase 2 测试 - 嵌入器与元数据库。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from retrievalhub.core.exceptions import DuplicateDocumentError
from retrievalhub.core.models import DocFormat, DocStatus, Document, KnowledgeBase
from retrievalhub.embedders.embedder import MockEmbedder
from retrievalhub.storage.metadata_store import MetadataStore


# ---- MockEmbedder 测试 ----


class TestMockEmbedder:
    @pytest.fixture
    def embedder(self):
        return MockEmbedder(dim=64)

    async def test_embed_returns_vectors(self, embedder):
        texts = ["hello", "world"]
        vectors = await embedder.embed(texts)
        assert len(vectors) == 2
        assert len(vectors[0]) == 64
        assert len(vectors[1]) == 64

    async def test_embed_query_single(self, embedder):
        vec = await embedder.embed_query("test query")
        assert len(vec) == 64

    async def test_embed_deterministic(self, embedder):
        """相同文本生成相同向量（确定性）。"""
        v1 = await embedder.embed_query("same text")
        v2 = await embedder.embed_query("same text")
        assert v1 == v2

    async def test_embed_different_text_different_vector(self, embedder):
        v1 = await embedder.embed_query("text A")
        v2 = await embedder.embed_query("text B")
        assert v1 != v2

    async def test_embed_l2_normalized(self, embedder):
        """向量应 L2 归一化。"""
        vec = await embedder.embed_query("normalized")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_dim_property(self, embedder):
        assert embedder.dim == 64


# ---- MetadataStore 测试 ----


class TestMetadataStore:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = MetadataStore(db_path=db_path)
        yield s
        s.close()

    def test_create_and_get_kb(self, store):
        kb = KnowledgeBase(id="kb1", name="Test KB", description="test")
        store.create_kb(kb)
        got = store.get_kb("kb1")
        assert got is not None
        assert got.name == "Test KB"
        assert got.description == "test"

    def test_get_nonexistent_kb(self, store):
        assert store.get_kb("nonexistent") is None

    def test_list_kbs(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        store.create_kb(KnowledgeBase(id="kb2", name="KB2"))
        kbs = store.list_kbs()
        assert len(kbs) == 2

    def test_delete_kb_cascades_documents(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        store.create_document(Document(
            id="doc1", kb_id="kb1", filename="test.md",
            file_format=DocFormat.MD, content_hash="abc",
        ))
        store.delete_kb("kb1")
        assert store.get_kb("kb1") is None
        assert store.get_document("doc1") is None

    def test_create_and_get_document(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        doc = Document(
            id="doc1", kb_id="kb1", filename="test.md",
            file_format=DocFormat.MD, file_size=100,
            content_hash="hash123",
        )
        store.create_document(doc)
        got = store.get_document("doc1")
        assert got is not None
        assert got.filename == "test.md"
        assert got.status == DocStatus.PROCESSING
        assert got.content_hash == "hash123"

    def test_duplicate_document_raises(self, store):
        """UNIQUE 约束 - 相同 (kb_id, content_hash) 返回 409。"""
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))

        doc1 = Document(
            id="doc1", kb_id="kb1", filename="test.md",
            file_format=DocFormat.MD, content_hash="same_hash",
        )
        store.create_document(doc1)

        doc2 = Document(
            id="doc2", kb_id="kb1", filename="dup.md",
            file_format=DocFormat.MD, content_hash="same_hash",
        )
        with pytest.raises(DuplicateDocumentError) as exc_info:
            store.create_document(doc2)
        assert exc_info.value.existing_doc_id == "doc1"

    def test_update_document_status(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        store.create_document(Document(
            id="doc1", kb_id="kb1", filename="test.md",
            file_format=DocFormat.MD, content_hash="h1",
        ))
        store.update_document_status(
            "doc1", DocStatus.READY, chunk_count=5
        )
        got = store.get_document("doc1")
        assert got.status == DocStatus.READY
        assert got.chunk_count == 5

    def test_update_status_failed_with_error(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        store.create_document(Document(
            id="doc1", kb_id="kb1", filename="test.md",
            file_format=DocFormat.MD, content_hash="h1",
        ))
        store.update_document_status(
            "doc1", DocStatus.FAILED, error_message="parse error"
        )
        got = store.get_document("doc1")
        assert got.status == DocStatus.FAILED
        assert got.error_message == "parse error"

    def test_list_documents_by_kb(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        for i in range(3):
            store.create_document(Document(
                id=f"doc{i}", kb_id="kb1", filename=f"test{i}.md",
                file_format=DocFormat.MD, content_hash=f"hash{i}",
            ))
        docs = store.list_documents("kb1")
        assert len(docs) == 3

    def test_list_documents_by_status(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        for i in range(3):
            store.create_document(Document(
                id=f"doc{i}", kb_id="kb1", filename=f"test{i}.md",
                file_format=DocFormat.MD, content_hash=f"hash{i}",
            ))
        store.update_document_status("doc0", DocStatus.READY)
        ready_docs = store.list_documents("kb1", status="ready")
        assert len(ready_docs) == 1
        assert ready_docs[0].id == "doc0"

    def test_delete_document(self, store):
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        store.create_document(Document(
            id="doc1", kb_id="kb1", filename="test.md",
            file_format=DocFormat.MD, content_hash="h1",
        ))
        store.delete_document("doc1")
        assert store.get_document("doc1") is None

    def test_attempt_count_increments(self, store):
        """重试时 attempt_count 应自增。"""
        store.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        store.create_document(Document(
            id="doc1", kb_id="kb1", filename="test.md",
            file_format=DocFormat.MD, content_hash="h1",
        ))
        # 初始 attempt_count = 0
        doc = store.get_document("doc1")
        assert doc.attempt_count == 0

        # 标记 processing（重试）
        store.update_document_status("doc1", DocStatus.PROCESSING)
        doc = store.get_document("doc1")
        assert doc.attempt_count == 1
        assert doc.started_at is not None
