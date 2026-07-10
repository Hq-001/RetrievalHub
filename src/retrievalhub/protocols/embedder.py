"""嵌入器协议 - 文本 -> 向量。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """嵌入模型接口。

    将文本（分块文本 / 查询）转换为向量。
    实现可插拔：OpenAI 兼容 / 本地模型（sentence-transformers 等）。
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本。

        Args:
            texts: 文本列表（分块文本）

        Returns:
            向量列表，维度由配置的 EMBEDDING_DIM 决定
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """嵌入单条查询文本。"""
        ...

    @property
    def dim(self) -> int:
        """向量维度。"""
        ...
