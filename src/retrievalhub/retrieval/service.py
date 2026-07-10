"""检索服务 - 编排三级管道：召回 -> 融合 -> 重排序。

无查询改写、无生成。
"""

from __future__ import annotations

import time

from retrievalhub.core.exceptions import SearchError
from retrievalhub.core.models import SearchHit, SearchRequest, SearchResponse
from retrievalhub.embedders.embedder import MockEmbedder
from retrievalhub.retrieval.fusion import DynamicWeightedFuser
from retrievalhub.retrieval.reranker import MockReranker
from retrievalhub.retrieval.recall import HybridRecaller
from retrievalhub.storage.lancedb_store import LanceDBStorage
from retrievalhub.storage.metadata_store import MetadataStore
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class SearchService:
    """检索服务编排器。

    执行：热点缓存命中 -> 混合召回 -> 动态加权融合 -> 重排序 -> 结构化组装。
    无查询改写、无 LLM 生成。
    """

    def __init__(
        self,
        storage: LanceDBStorage,
        metadata: MetadataStore,
        embedder: MockEmbedder | None = None,
        recaller: HybridRecaller | None = None,
        fuser: DynamicWeightedFuser | None = None,
        reranker: MockReranker | None = None,
        md_vector_weight: float = 1.1,
        json_bm25_weight: float = 1.2,
        vector_top_n: int = 50,
        bm25_top_n: int = 50,
        rerank_top_k: int = 10,
    ) -> None:
        self._storage = storage
        self._metadata = metadata
        self._embedder = embedder or MockEmbedder()

        self._recaller = recaller or HybridRecaller(
            storage, vector_top_n=vector_top_n, bm25_top_n=bm25_top_n
        )
        self._fuser = fuser or DynamicWeightedFuser(
            md_vector_weight=md_vector_weight,
            json_bm25_weight=json_bm25_weight,
        )
        self._reranker = reranker or MockReranker(enabled=True, top_k=rerank_top_k)
        self._rerank_top_k = rerank_top_k

    async def search(self, request: SearchRequest) -> SearchResponse:
        """执行检索。

        Args:
            request: 检索请求

        Returns:
            SearchResponse，含结构化 Chunk 列表（不含生成内容）
        """
        start = time.time()
        kb = self._metadata.get_kb(request.kb_id)

        if kb is None:
            raise SearchError(f"知识库不存在: {request.kb_id}")

        collection = kb.active_collection
        if not collection:
            raise SearchError(f"知识库 {request.kb_id} 无活跃集合")

        # 1. 嵌入查询
        query_vector = await self._embedder.embed_query(request.query)

        # 2. 混合召回（向量 + BM25 并行）
        vector_hits, bm25_hits = await self._recaller.recall(
            collection, request.query, query_vector, request.filters
        )

        # 3. 动态加权融合
        fused = self._fuser.fuse(vector_hits, bm25_hits, top_n=max(len(vector_hits), len(bm25_hits)))

        # 4. 重排序（可选）
        if request.rerank and self._reranker.is_enabled:
            final = await self._reranker.rerank(
                request.query, fused, top_k=request.top_k
            )
        else:
            final = fused[: request.top_k]

        # 5. 组装结构化结果
        hits = []
        for item in final:
            # 查找文档名
            doc = self._metadata.get_document(item.get("doc_id", ""))
            doc_name = doc.filename if doc else ""

            hits.append(
                SearchHit(
                    chunk_id=item.get("chunk_id", ""),
                    text=item.get("text", ""),
                    score=float(item.get("rerank_score", item.get("fused_score", 0.0))),
                    content_type=item.get("content_type", "md"),
                    section_path=item.get("section_path", ""),
                    json_parent_id=item.get("json_parent_id") or None,
                    code_language=item.get("code_language") or None,
                    char_offset=int(item.get("char_offset", 0)),
                    doc_id=item.get("doc_id", ""),
                    doc_name=doc_name,
                )
            )

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            "search_complete",
            query=request.query,
            kb_id=request.kb_id,
            total_hits=len(hits),
            elapsed_ms=elapsed_ms,
            rerank_used=request.rerank and self._reranker.is_enabled,
        )

        return SearchResponse(
            hits=hits,
            total=len(hits),
            cached=False,
        )
