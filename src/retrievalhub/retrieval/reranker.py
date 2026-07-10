"""重排序器 - Cross-encoder 精排 + Mock 实现。

仅作用于检索结果排序，绝不参与生成。
可通过配置关闭以降低延迟。
"""

from __future__ import annotations

import hashlib
from typing import Any

from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class MockReranker:
    """Mock 重排序器 - 用于测试和无 API 场景。

    基于查询与文本的词频重叠模拟 cross-encoder 打分，
    保证测试可复现且有意义。
    """

    def __init__(self, enabled: bool = True, top_k: int = 20) -> None:
        self._enabled = enabled
        self._top_k = top_k

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """对候选结果重排序。

        Args:
            query: 查询文本
            candidates: 融合后的候选列表（含 text 字段）
            top_k: 返回数量上限

        Returns:
            重排序后的候选列表（含 rerank_score），按分数降序
        """
        if not self._enabled or not candidates:
            return candidates[:top_k]

        query_words = set(query.lower().split())

        for candidate in candidates:
            text = candidate.get("text", "")
            text_words = set(text.lower().split())

            # 词频重叠率作为模拟相关性分数
            if query_words:
                overlap = len(query_words & text_words)
                score = overlap / len(query_words)
            else:
                score = 0.0

            # 加上融合分数作为基础（避免同等词频时乱序）
            base = candidate.get("fused_score", 0.0)
            candidate["rerank_score"] = score + base * 0.1

        result = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        logger.info(
            "rerank_complete",
            input_count=len(candidates),
            output_count=min(len(result), top_k),
        )

        return result[:top_k]

    @property
    def is_enabled(self) -> bool:
        return self._enabled


class DisabledReranker:
    """禁用的重排序器 - 直接透传（降低延迟用）。"""

    def __init__(self) -> None:
        self._enabled = False

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        return candidates[:top_k]

    @property
    def is_enabled(self) -> bool:
        return False
