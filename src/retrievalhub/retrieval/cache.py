"""热点查询 LRU 缓存 - 精细失效（doc_versions_hash 分区驱逐）。

缓存键: (kb_id, doc_versions_hash, query, params)
文档变更时仅更新 doc_versions_hash，使仅依赖被改文档的查询结果因哈希失配而自动驱逐。
兜底方案：TTL 为主 + 主动失效为辅。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class LRUCache:
    """带 TTL 的 LRU 缓存。

    支持精细失效：基于 doc_versions_hash 分区驱逐，
    单文档更新不清空整个知识库命名空间。
    """

    def __init__(self, maxsize: int = 512, ttl_sec: int = 300) -> None:
        self._maxsize = maxsize
        self._ttl_sec = ttl_sec
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._kb_versions: dict[str, str] = {}  # kb_id -> doc_versions_hash
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        kb_id: str,
        query: str,
        params: dict | None,
        doc_versions_hash: str | None = None,
    ) -> str:
        """生成缓存键。"""
        if doc_versions_hash is None:
            doc_versions_hash = self._kb_versions.get(kb_id, "default")
        params_str = json.dumps(params or {}, sort_keys=True)
        return f"{kb_id}:{doc_versions_hash}:{query}:{params_str}"

    def compute_doc_versions_hash(self, kb_id: str, doc_versions: list[tuple[str, str]]) -> str:
        """计算知识库下所有 (doc_id, last_modified) 的全局哈希。

        Args:
            kb_id: 知识库 ID
            doc_versions: [(doc_id, last_modified), ...]

        Returns:
            doc_versions_hash 字符串
        """
        data = json.dumps(sorted(doc_versions), sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def update_kb_version(self, kb_id: str, doc_versions_hash: str) -> None:
        """更新知识库的 doc_versions_hash（文档变更时调用）。

        更新后，旧哈希的缓存键自然失配，实现分区驱逐。
        """
        old = self._kb_versions.get(kb_id)
        self._kb_versions[kb_id] = doc_versions_hash
        if old and old != doc_versions_hash:
            # 清理该 kb_id 下旧哈希的缓存项
            keys_to_evict = [
                k for k in self._cache
                if k.startswith(f"{kb_id}:{old}:")
            ]
            for k in keys_to_evict:
                self._cache.pop(k, None)
            logger.info(
                "cache_version_evicted",
                kb_id=kb_id,
                evicted_count=len(keys_to_evict),
            )

    def get(self, kb_id: str, query: str, params: dict | None = None) -> Any | None:
        """查询缓存。"""
        key = self._make_key(kb_id, query, params)
        if key not in self._cache:
            self._misses += 1
            return None

        value, timestamp = self._cache[key]
        # TTL 检查
        if time.time() - timestamp > self._ttl_sec:
            self._cache.pop(key)
            self._misses += 1
            return None

        # LRU: 移到末尾
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(
        self,
        kb_id: str,
        query: str,
        value: Any,
        params: dict | None = None,
    ) -> None:
        """写入缓存。"""
        key = self._make_key(kb_id, query, params)
        self._cache[key] = (value, time.time())
        self._cache.move_to_end(key)

        # 容量淘汰
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def invalidate_kb(self, kb_id: str) -> int:
        """失效整个知识库的缓存。"""
        keys_to_evict = [k for k in self._cache if k.startswith(f"{kb_id}:")]
        for k in keys_to_evict:
            self._cache.pop(k)
        self._kb_versions.pop(kb_id, None)
        return len(keys_to_evict)

    @property
    def stats(self) -> dict:
        """缓存统计。"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """清空全部缓存。"""
        self._cache.clear()
        self._kb_versions.clear()
        self._hits = 0
        self._misses = 0
