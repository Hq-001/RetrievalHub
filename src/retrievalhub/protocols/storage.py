"""存储协议 - 向量 + 全文索引持久化与混合检索。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrievalhub.core.models import Chunk


@runtime_checkable
class Storage(Protocol):
    """存储接口。

    持久化向量与全文索引，支持高效混合检索。
    默认实现 LanceDB（原生向量 + FTS/BM25 于同一存储）。
    可替换为 Milvus / Qdrant + 独立 FTS。
    """

    async def create_collection(
        self, kb_id: str, dim: int
    ) -> str:
        """创建知识库对应的集合。

        Args:
            kb_id: 知识库 ID
            dim: 向量维度

        Returns:
            集合名（collection name）
        """
        ...

    async def delete_collection(self, collection: str) -> None:
        """删除整个集合（删除知识库时调用）。"""
        ...

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """写入/更新分块与向量。

        Args:
            collection: 集合名
            chunks: 分块列表
            vectors: 对应的向量列表（顺序一致）
        """
        ...

    async def delete_by_doc(self, collection: str, doc_id: str) -> None:
        """删除指定文档的全部分块与向量。"""
        ...

    async def vector_search(
        self,
        collection: str,
        query_vector: list[float],
        top_n: int,
        filters: dict | None = None,
    ) -> list[dict]:
        """向量召回。

        Returns:
            候选结果列表，每项含 chunk 元信息与相似度得分
        """
        ...

    async def fts_search(
        self,
        collection: str,
        query: str,
        top_n: int,
        filters: dict | None = None,
    ) -> list[dict]:
        """全文召回（BM25）。

        Returns:
            候选结果列表，每项含 chunk 元信息与 BM25 得分
        """
        ...

    async def health_check(self) -> bool:
        """存储健康检查（供 /readyz）。"""
        ...
