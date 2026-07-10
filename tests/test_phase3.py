"""Phase 3 测试 - 异步入库流水线、崩溃恢复。"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from retrievalhub.core.exceptions import DuplicateDocumentError, IngestError
from retrievalhub.core.models import DocFormat, DocStatus, Document, KnowledgeBase
from retrievalhub.embedders.embedder import MockEmbedder
from retrievalhub.ingest.crash_recovery import CrashRecovery
from retrievalhub.ingest.enqueue import InProcessEnqueueBackend
from retrievalhub.ingest.file_handler import (
    compute_content_hash,
    detect_format,
    read_file_content,
    save_original_file,
    validate_content,
)
from retrievalhub.ingest.pipeline import IngestPipeline
from retrievalhub.storage.lancedb_store import LanceDBStorage
from retrievalhub.storage.metadata_store import MetadataStore

FIXTURES = Path(__file__).parent / "fixtures"


# ---- 文件处理测试 ----


class TestFileHandler:
    def test_detect_format_md(self):
        assert detect_format("test.md") == DocFormat.MD

    def test_detect_format_json(self):
        assert detect_format("test.json") == DocFormat.JSON

    def test_detect_format_jsonl(self):
        assert detect_format("test.jsonl") == DocFormat.JSONL

    def test_detect_format_unsupported(self):
        with pytest.raises(Exception):
            detect_format("test.pdf")

    def test_validate_content_md(self):
        validate_content("# Title\nContent.", DocFormat.MD)

    def test_validate_content_json_valid(self):
        validate_content('{"key": "value"}', DocFormat.JSON)

    def test_validate_content_json_invalid(self):
        with pytest.raises(Exception):
            validate_content("{invalid}", DocFormat.JSON)

    def test_validate_content_jsonl_valid(self):
        validate_content('{"a": 1}\n{"b": 2}', DocFormat.JSONL)

    def test_validate_content_empty(self):
        with pytest.raises(Exception):
            validate_content("", DocFormat.MD)

    def test_compute_content_hash_stable(self):
        h1 = compute_content_hash("same content")
        h2 = compute_content_hash("same content")
        assert h1 == h2

    def test_compute_content_hash_different(self):
        h1 = compute_content_hash("content A")
        h2 = compute_content_hash("content B")
        assert h1 != h2

    def test_compute_content_hash_is_sha256(self):
        h = compute_content_hash("test")
        assert len(h) == 64  # SHA-256 hex length

    def test_save_and_read_file(self, tmp_path):
        content = "# Test\nContent."
        path = save_original_file(content, "test.md", tmp_path)
        assert path.exists()
        read_back = read_file_content(path)
        assert read_back == content


# ---- 异步后端测试 ----


class TestEnqueueBackend:
    @pytest.fixture
    def backend(self):
        return InProcessEnqueueBackend(timeout_sec=10)

    async def test_submit_and_complete(self, backend):
        async def sample_task():
            await asyncio.sleep(0.01)
            return "done"

        task_id = await backend.submit(sample_task)
        await asyncio.sleep(0.1)
        assert backend.get_status(task_id) == "completed"

    async def test_submit_with_task_id(self, backend):
        async def task():
            return 42

        tid = await backend.submit(task, task_id="custom-id")
        assert tid == "custom-id"
        await asyncio.sleep(0.05)
        assert backend.get_status("custom-id") == "completed"

    async def test_timeout_marks_failed(self, backend):
        backend = InProcessEnqueueBackend(timeout_sec=1)

        async def slow_task():
            await asyncio.sleep(5)
            return "should not reach"

        task_id = await backend.submit(slow_task)
        await asyncio.sleep(2)
        assert backend.get_status(task_id) == "failed"

    async def test_exception_marks_failed(self, backend):
        async def failing_task():
            raise ValueError("task error")

        task_id = await backend.submit(failing_task)
        await asyncio.sleep(0.1)
        assert backend.get_status(task_id) == "failed"

    async def test_pending_tasks(self, backend):
        async def task():
            await asyncio.sleep(0.5)
            return None

        await backend.submit(task, task_id="t1")
        await backend.submit(task, task_id="t2")
        pending = backend.pending_tasks()
        assert "t1" in pending
        assert "t2" in pending
        await asyncio.sleep(1)
        assert backend.pending_tasks() == []

    async def test_wait_for_completion(self, backend):
        async def task():
            await asyncio.sleep(0.05)
            return "result"

        tid = await backend.submit(task)
        result = await backend.wait_for_completion(tid, timeout=5)
        assert result == "result"


# ---- 入库流水线测试 ----


class TestIngestPipeline:
    @pytest.fixture
    def setup(self, tmp_path):
        storage = LanceDBStorage(
            uri=str(tmp_path / "lancedb"),
            distance_metric="cosine",
        )
        metadata = MetadataStore(db_path=str(tmp_path / "metadata.db"))
        embedder = MockEmbedder(dim=64)
        pipeline = IngestPipeline(
            storage=storage,
            metadata=metadata,
            embedder=embedder,
        )
        yield storage, metadata, pipeline, embedder
        metadata.close()

    async def test_ingest_md_file(self, setup):
        storage, metadata, pipeline, embedder = setup

        # 准备知识库
        kb_id = "kb-test"
        metadata.create_kb(KnowledgeBase(id=kb_id, name="Test"))
        collection = await storage.create_collection(kb_id, dim=embedder.dim)

        # 准备文档
        content = (FIXTURES / "sample.md").read_text(encoding="utf-8")
        file_path = FIXTURES / "sample.md"
        content_hash = compute_content_hash(content)

        doc = Document(
            id="doc-md-1",
            kb_id=kb_id,
            filename="sample.md",
            file_format=DocFormat.MD,
            content_hash=content_hash,
        )
        metadata.create_document(doc)

        # 执行入库
        await pipeline.ingest("doc-md-1", file_path, kb_id, content_hash, collection)

        # 验证状态
        doc = metadata.get_document("doc-md-1")
        assert doc.status == DocStatus.READY
        assert doc.chunk_count > 0

    async def test_ingest_json_file(self, setup):
        storage, metadata, pipeline, embedder = setup

        kb_id = "kb-json"
        metadata.create_kb(KnowledgeBase(id=kb_id, name="JSON Test"))
        collection = await storage.create_collection(kb_id, dim=embedder.dim)

        content = (FIXTURES / "sample.json").read_text(encoding="utf-8")
        file_path = FIXTURES / "sample.json"
        content_hash = compute_content_hash(content)

        doc = Document(
            id="doc-json-1",
            kb_id=kb_id,
            filename="sample.json",
            file_format=DocFormat.JSON,
            content_hash=content_hash,
        )
        metadata.create_document(doc)

        await pipeline.ingest("doc-json-1", file_path, kb_id, content_hash, collection)

        doc = metadata.get_document("doc-json-1")
        assert doc.status == DocStatus.READY
        assert doc.chunk_count > 0

    async def test_ingest_jsonl_file(self, setup):
        storage, metadata, pipeline, embedder = setup

        kb_id = "kb-jsonl"
        metadata.create_kb(KnowledgeBase(id=kb_id, name="JSONL Test"))
        collection = await storage.create_collection(kb_id, dim=embedder.dim)

        content = (FIXTURES / "sample.jsonl").read_text(encoding="utf-8")
        file_path = FIXTURES / "sample.jsonl"
        content_hash = compute_content_hash(content)

        doc = Document(
            id="doc-jsonl-1",
            kb_id=kb_id,
            filename="sample.jsonl",
            file_format=DocFormat.JSONL,
            content_hash=content_hash,
        )
        metadata.create_document(doc)

        await pipeline.ingest("doc-jsonl-1", file_path, kb_id, content_hash, collection)

        doc = metadata.get_document("doc-jsonl-1")
        assert doc.status == DocStatus.READY
        assert doc.chunk_count > 0

    async def test_ingest_failure_marks_failed(self, setup):
        storage, metadata, pipeline, embedder = setup

        kb_id = "kb-fail"
        metadata.create_kb(KnowledgeBase(id=kb_id, name="Fail Test"))
        collection = await storage.create_collection(kb_id, dim=embedder.dim)

        # 写一个无效文件
        bad_file = Path(tempfile.mktemp(suffix=".json"))
        bad_file.write_text("{invalid json}", encoding="utf-8")

        doc = Document(
            id="doc-fail-1",
            kb_id=kb_id,
            filename="bad.json",
            file_format=DocFormat.JSON,
            content_hash="hash",
        )
        metadata.create_document(doc)

        with pytest.raises(Exception):
            await pipeline.ingest("doc-fail-1", bad_file, kb_id, "hash", collection)

        doc = metadata.get_document("doc-fail-1")
        assert doc.status == DocStatus.FAILED
        assert doc.error_message is not None

        bad_file.unlink()


# ---- 崩溃恢复测试 ----


class TestCrashRecovery:
    @pytest.fixture
    def setup(self, tmp_path):
        metadata = MetadataStore(db_path=str(tmp_path / "metadata.db"))
        recovery = CrashRecovery(metadata, timeout_sec=0)
        yield metadata, recovery
        metadata.close()

    def test_no_stale_docs(self, setup):
        metadata, recovery = setup
        metadata.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        count = recovery.recover_stale_documents()
        assert count == 0

    def test_recover_stale_doc(self, setup):
        metadata, recovery = setup
        metadata.create_kb(KnowledgeBase(id="kb1", name="KB1"))

        # 创建一个 processing 文档（started_at 为很久以前）
        doc = Document(
            id="doc-stale",
            kb_id="kb1",
            filename="old.md",
            file_format=DocFormat.MD,
            content_hash="h1",
        )
        metadata.create_document(doc)
        metadata.update_document_status("doc-stale", DocStatus.PROCESSING)

        # 等待使其超时
        import time

        time.sleep(0.1)

        # 使用 0 秒超时 -> 立即标记为 stale
        recovery = CrashRecovery(metadata, timeout_sec=0)
        count = recovery.recover_stale_documents()
        assert count == 1

        recovered_doc = metadata.get_document("doc-stale")
        assert recovered_doc.status == DocStatus.FAILED
        assert recovered_doc.error_message == "process_restarted"

    def test_does_not_recover_ready_doc(self, setup):
        metadata, recovery = setup
        metadata.create_kb(KnowledgeBase(id="kb1", name="KB1"))

        doc = Document(
            id="doc-ready",
            kb_id="kb1",
            filename="ok.md",
            file_format=DocFormat.MD,
            content_hash="h1",
        )
        metadata.create_document(doc)
        metadata.update_document_status("doc-ready", DocStatus.READY, chunk_count=5)

        recovery = CrashRecovery(metadata, timeout_sec=0)
        count = recovery.recover_stale_documents()
        assert count == 0
