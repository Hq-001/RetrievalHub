"""重排序器协议 - Cross-encoder 精排。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """重排序器接口。

    对融合候选调用 cross-encoder 做 query-chunk 精排。
    仅作用于检索结果排序，绝不参与生成。
    可通过 .env 关闭以降低延迟。
    """

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """对候选结果重排序。

        Args:
            query: 查询文本
            candidates: 融合后的候选列表（含 text 字段）
            top_k: 返回数量上限

        Returns:
            重排序后的候选列表（含 rerank_score）
        """
        ...

    @property
    def is_enabled(self) -> bool:
        """是否启用重排序。"""
        ...
