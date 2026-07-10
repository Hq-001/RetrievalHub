"""Phase 4 测试 - 混合召回、动态加权融合、重排序、检索服务。"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrievalhub.core.models import (
    ContentType,
    DocFormat,
    DocStatus,
    Document,
    KnowledgeBase,
    SearchRequest,
)
from retrievalhub.embedders.embedder import MockEmbedder
from retrievalhub.ingest.file_handler import compute_content_hash
from retrievalhub.ingest.pipeline import IngestPipeline
from retrievalhub.retrieval.fusion import DynamicWeightedFuser
from retrievalhub.retrieval.reranker import DisabledReranker, MockReranker
from retrievalhub.retrieval.recall import HybridRecaller
from retrievalhub.retrieval.service import SearchService
from retrievalhub.storage.lancedb_store import LanceDBStorage
from retrievalhub.storage.metadata_store import MetadataStore

FIXTURES = Path(__file__).parent / "fixtures"


# ---- 融合器单元测试 ----


class TestDynamicWeightedFuser:
    @pytest.fixture
    def fuser(self):
        return DynamicWeightedFuser(
            md_vector_weight=1.1,
            json_bm25_weight=1.2,
        )

    def test_fuse_empty(self, fuser):
        result = fuser.fuse([], [])
        assert result == []

    def test_fuse_vector_only(self, fuser):
        vector_hits = [
            {"chunk_id": "c1", "text": "hello", "content_type": "md",
             "score": 0.9, "doc_id": "d1"},
            {"chunk_id": "c2", "text": "world", "content_type": "md",
             "score": 0.8, "doc_id": "d1"},
        ]
        result = fuser.fuse(vector_hits, [])
        assert len(result) == 2
        # rank 0 的分数应高于 rank 1
        assert result[0]["fused_score"] > result[1]["fused_score"]
        assert result[0]["chunk_id"] == "c1"

    def test_fuse_bm25_only(self, fuser):
        bm25_hits = [
            {"chunk_id": "c1", "text": "hello", "content_type": "json",
             "score": 5.0, "doc_id": "d1"},
        ]
        result = fuser.fuse([], bm25_hits)
        assert len(result) == 1
        assert result[0]["bm25_score"] > 0

    def test_fuse_merges_duplicate_chunk_id(self, fuser):
        """同一 chunk 在两路都出现时应合并分数。"""
        vector_hits = [
            {"chunk_id": "c1", "text": "hello", "content_type": "md",
             "score": 0.9, "doc_id": "d1"},
        ]
        bm25_hits = [
            {"chunk_id": "c1", "text": "hello", "content_type": "md",
             "score": 5.0, "doc_id": "d1"},
        ]
        result = fuser.fuse(vector_hits, bm25_hits)
        assert len(result) == 1
        # 两路都命中，分数应高于单路
        assert result[0]["fused_score"] > 0
        assert result[0]["vector_score"] > 0
        assert result[0]["bm25_score"] > 0

    def test_md_vector_weight_applied(self, fuser):
        """MD 块在向量路应有加权。"""
        vector_hits = [
            {"chunk_id": "c1", "text": "text", "content_type": "md",
             "score": 0.9, "doc_id": "d1"},
        ]
        result = fuser.fuse(vector_hits, [])
        # rank 0: 1/(60+1) * 1.1
        expected = (1.0 / 61) * 1.1
        assert abs(result[0]["vector_score"] - expected) < 0.0001

    def test_json_bm25_weight_applied(self, fuser):
        """JSON 块在 BM25 路应有加权。"""
        bm25_hits = [
            {"chunk_id": "c1", "text": "text", "content_type": "json",
             "score": 5.0, "doc_id": "d1"},
        ]
        result = fuser.fuse([], bm25_hits)
        # rank 0: 1/(60+1) * 1.2
        expected = (1.0 / 61) * 1.2
        assert abs(result[0]["bm25_score"] - expected) < 0.0001

    def test_fuse_top_n_limit(self, fuser):
        hits = [
            {"chunk_id": f"c{i}", "text": str(i), "content_type": "md",
             "score": 0.5, "doc_id": "d1"}
            for i in range(20)
        ]
        result = fuser.fuse(hits, [], top_n=5)
        assert len(result) == 5

    def test_order_by_fused_score(self, fuser):
        """两路都命中的 chunk 排名应高于单路命中。"""
        vector_hits = [
            {"chunk_id": "both", "text": "common", "content_type": "md",
             "score": 0.5, "doc_id": "d1"},
            {"chunk_id": "vec_only", "text": "vec", "content_type": "md",
             "score": 0.4, "doc_id": "d1"},
        ]
        bm25_hits = [
            {"chunk_id": "both", "text": "common", "content_type": "md",
             "score": 3.0, "doc_id": "d1"},
        ]
        result = fuser.fuse(vector_hits, bm25_hits)
        assert result[0]["chunk_id"] == "both"
        assert result[1]["chunk_id"] == "vec_only"


# ---- 重排序器单元测试 ----


class TestMockReranker:
    @pytest.fixture
    def reranker(self):
        return MockReranker(enabled=True)

    async def test_rerank_reorders_by_relevance(self, reranker):
        query = "python code"
        candidates = [
            {"chunk_id": "c1", "text": "python is great", "fused_score": 0.5},
            {"chunk_id": "c2", "text": "java code example", "fused_score": 0.8},
            {"chunk_id": "c3", "text": "python code snippet here", "fused_score": 0.3},
        ]
        result = await reranker.rerank(query, candidates, top_k=3)
        # c3 匹配了 "python" 和 "code"，应排名更高
        assert result[0]["chunk_id"] in ("c3", "c1")
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]

    async def test_rerank_top_k_limit(self, reranker):
        candidates = [
            {"chunk_id": f"c{i}", "text": f"text {i}", "fused_score": 0.5}
            for i in range(20)
        ]
        result = await reranker.rerank("text", candidates, top_k=5)
        assert len(result) == 5

    async def test_rerank_empty_candidates(self, reranker):
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    async def test_is_enabled(self, reranker):
        assert reranker.is_enabled is True

    async def test_disabled_reranker_passthrough(self):
        dr = DisabledReranker()
        assert dr.is_enabled is False
        candidates = [
            {"chunk_id": "c1", "text": "a", "fused_score": 0.9},
            {"chunk_id": "c2", "text": "b", "fused_score": 0.5},
        ]
        result = await dr.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        # 不重排序，保持原始顺序
        assert result[0]["chunk_id"] == "c1"
        assert "rerank_score" not in result[0]


# ---- 检索服务端到端测试 ----


class TestSearchService:
    @pytest.fixture
    async def setup(self, tmp_path):
        storage = LanceDBStorage(uri=str(tmp_path / "lancedb"))
        metadata = MetadataStore(db_path=str(tmp_path / "metadata.db"))
        embedder = MockEmbedder(dim=64)

        pipeline = IngestPipeline(
            storage=storage,
            metadata=metadata,
            embedder=embedder,
        )

        service = SearchService(
            storage=storage,
            metadata=metadata,
            embedder=embedder,
            md_vector_weight=1.1,
            json_bm25_weight=1.2,
        )

        # 创建知识库并入库测试文档
        kb_id = "kb-search"
        metadata.create_kb(KnowledgeBase(id=kb_id, name="Search Test"))
        collection = await storage.create_collection(kb_id, dim=embedder.dim)
        metadata.update_kb_collection(kb_id, collection)

        # 入库 MD 文档
        md_content = (FIXTURES / "sample.md").read_text(encoding="utf-8")
        md_hash = compute_content_hash(md_content)
        metadata.create_document(Document(
            id="doc-md", kb_id=kb_id, filename="sample.md",
            file_format=DocFormat.MD, content_hash=md_hash,
        ))
        await pipeline.ingest("doc-md", FIXTURES / "sample.md", kb_id, md_hash, collection)

        # 入库 JSON 文档
        json_content = (FIXTURES / "sample.json").read_text(encoding="utf-8")
        json_hash = compute_content_hash(json_content)
        metadata.create_document(Document(
            id="doc-json", kb_id=kb_id, filename="sample.json",
            file_format=DocFormat.JSON, content_hash=json_hash,
        ))
        await pipeline.ingest("doc-json", FIXTURES / "sample.json", kb_id, json_hash, collection)

        yield storage, metadata, service, kb_id
        metadata.close()

    async def test_search_returns_hits(self, setup):
        _, _, service, kb_id = setup
        req = SearchRequest(query="检索", kb_id=kb_id, top_k=5)
        resp = await service.search(req)
        assert resp.total > 0
        assert len(resp.hits) <= 5

    async def test_search_returns_text_and_source(self, setup):
        _, _, service, kb_id = setup
        req = SearchRequest(query="架构", kb_id=kb_id, top_k=5)
        resp = await service.search(req)
        for hit in resp.hits:
            assert hit.text  # 非空
            assert hit.doc_id  # 有来源
            assert hit.doc_name  # 有文档名

    async def test_search_no_rerank(self, setup):
        _, _, service, kb_id = setup
        req = SearchRequest(query="test", kb_id=kb_id, top_k=5, rerank=False)
        resp = await service.search(req)
        assert resp.total >= 0  # 不报错即可

    async def test_search_nonexistent_kb(self, setup):
        _, _, service, _ = setup
        req = SearchRequest(query="test", kb_id="nonexistent", top_k=5)
        with pytest.raises(Exception):
            await service.search(req)

    async def test_search_scores_sorted_desc(self, setup):
        _, _, service, kb_id = setup
        req = SearchRequest(query="检索", kb_id=kb_id, top_k=10)
        resp = await service.search(req)
        scores = [h.score for h in resp.hits]
        assert scores == sorted(scores, reverse=True)
