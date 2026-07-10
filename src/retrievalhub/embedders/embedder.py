"""嵌入模型适配器 - OpenAI 兼容协议 + Mock 实现。"""

from __future__ import annotations

import asyncio
import hashlib
import random

import httpx

from retrievalhub.core.exceptions import EmbeddingError
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class MockEmbedder:
    """Mock 嵌入器 - 用于测试和本地无 API 场景。

    生成确定性哈希向量（相同文本 -> 相同向量），
    保证测试可复现。
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed_one(text)

    async def _embed_one(self, text: str) -> list[float]:
        """确定性哈希向量生成。"""
        vec = [0.0] * self._dim
        for i in range(self._dim):
            h = hashlib.md5(f"{text}:{i}".encode()).hexdigest()
            vec[i] = (int(h[:8], 16) % 1000) / 1000.0
        # L2 归一化
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @property
    def dim(self) -> int:
        return self._dim


class OpenAICompatibleEmbedder:
    """OpenAI 兼容协议嵌入器。

    通过 .env 配置 Endpoint / Key / 模型名 / 维度 / 批量大小。
    不绑定特定厂商，OpenAI 兼容协议即可。
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        batch_size: int = 64,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入，自动分批。"""
        results: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_vecs = await self._embed_batch(batch)
            results.extend(batch_vecs)

        return results

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """单批嵌入调用，含指数退避重试。"""
        client = await self._ensure_client()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    "/v1/embeddings",
                    json={
                        "model": self._model,
                        "input": texts,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 0.2 * (2 ** attempt)
                    logger.warning(
                        "embed_retry",
                        attempt=attempt + 1,
                        wait=wait,
                        error=str(e),
                    )
                    await asyncio.sleep(wait)
                else:
                    raise EmbeddingError(f"嵌入调用失败（重试 {max_retries} 次）: {e}") from e

        return []  # unreachable

    @property
    def dim(self) -> int:
        return self._dim

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
