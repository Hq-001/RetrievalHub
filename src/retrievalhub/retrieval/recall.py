"""混合召回器 - 向量召回 + 全文(BM25)召回并行。"""

from __future__ import annotations

import asyncio

from retrievalhub.storage.lancedb_store import LanceDBStorage
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRecaller:
    """混合召回器。

    向量召回（语义相似度）+ 全文召回（BM25）并行执行，
    各取 top-N 候选进入融合阶段。
    """

    def __init__(
        self,
        storage: LanceDBStorage,
        vector_top_n: int = 50,
        bm25_top_n: int = 50,
    ) -> None:
        self._storage = storage
        self._vector_top_n = vector_top_n
        self._bm25_top_n = bm25_top_n

    async def recall(
        self,
        collection: str,
        query: str,
        query_vector: list[float],
        filters: dict | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """并行执行向量召回与全文召回。

        Args:
            collection: 集合名
            query: 查询文本（BM25 用）
            query_vector: 查询向量
            filters: 元信息过滤

        Returns:
            (vector_hits, bm25_hits) 两路候选列表
        """
        vector_task = self._storage.vector_search(
            collection, query_vector, self._vector_top_n, filters
        )
        bm25_task = self._storage.fts_search(
            collection, query, self._bm25_top_n, filters
        )

        vector_hits, bm25_hits = await asyncio.gather(
            vector_task, bm25_task, return_exceptions=True
        )

        # 容错：某一路失败不阻塞另一路
        if isinstance(vector_hits, Exception):
            logger.warning("vector_recall_failed", error=str(vector_hits))
            vector_hits = []
        if isinstance(bm25_hits, Exception):
            logger.warning("bm25_recall_failed", error=str(bm25_hits))
            bm25_hits = []

        logger.info(
            "recall_complete",
            vector_count=len(vector_hits),
            bm25_count=len(bm25_hits),
        )
        return vector_hits, bm25_hits
