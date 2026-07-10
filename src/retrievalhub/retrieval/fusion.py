"""动态加权融合器 - RRF 基础 + 按 content_type 差异化加权。

核心策略：
- 基础融合采用 RRF（Reciprocal Rank Fusion）
- MD 块：向量得分 × MD_VECTOR_WEIGHT（默认 1.1）--叙述性文本语义检索更有效
- JSON 块：BM25 得分 × JSON_BM25_WEIGHT（默认 1.2）--键值/编号精确匹配更有效
"""

from __future__ import annotations

from retrievalhub.core.models import ContentType
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)

# RRF 超参数（标准值 60）
RRF_K = 60


class DynamicWeightedFuser:
    """按 content_type 动态加权融合器。

    将向量召回与 BM25 召回的结果按 RRF 融合，
    并根据 content_type 对不同召回路径施加差异化权重。
    """

    def __init__(
        self,
        md_vector_weight: float = 1.1,
        json_bm25_weight: float = 1.2,
    ) -> None:
        self._md_vector_weight = md_vector_weight
        self._json_bm25_weight = json_bm25_weight

    def fuse(
        self,
        vector_hits: list[dict],
        bm25_hits: list[dict],
        top_n: int = 100,
    ) -> list[dict]:
        """融合两路召回结果。

        Args:
            vector_hits: 向量召回候选列表
            bm25_hits: BM25 召回候选列表
            top_n: 融合后返回数量上限

        Returns:
            融合后的候选列表（含 fused_score），按分数降序
        """
        # 构建 chunk_id -> 候选信息的映射
        merged: dict[str, dict] = {}

        # 向量召回：按 rank 计算 RRF 分
        for rank, hit in enumerate(vector_hits):
            chunk_id = hit["chunk_id"]
            rrf_score = 1.0 / (RRF_K + rank + 1)

            # 按 content_type 动态加权
            ct = hit.get("content_type", "md")
            if ct == ContentType.MD.value:
                rrf_score *= self._md_vector_weight
            # JSON 块在向量路不额外加权（其优势在 BM25 路）

            if chunk_id in merged:
                merged[chunk_id]["vector_score"] = rrf_score
            else:
                merged[chunk_id] = {
                    **hit,
                    "vector_score": rrf_score,
                    "bm25_score": 0.0,
                }

        # BM25 召回：按 rank 计算 RRF 分
        for rank, hit in enumerate(bm25_hits):
            chunk_id = hit["chunk_id"]
            rrf_score = 1.0 / (RRF_K + rank + 1)

            # 按 content_type 动态加权
            ct = hit.get("content_type", "md")
            if ct == ContentType.JSON.value:
                rrf_score *= self._json_bm25_weight
            # MD 块在 BM25 路不额外加权（其优势在向量路）

            if chunk_id in merged:
                merged[chunk_id]["bm25_score"] = rrf_score
            else:
                merged[chunk_id] = {
                    **hit,
                    "vector_score": 0.0,
                    "bm25_score": rrf_score,
                }

        # 融合：vector_score + bm25_score
        for chunk_id, item in merged.items():
            item["fused_score"] = item["vector_score"] + item["bm25_score"]

        # 按融合分数降序排列
        result = sorted(
            merged.values(),
            key=lambda x: x["fused_score"],
            reverse=True,
        )

        logger.info(
            "fuse_complete",
            input_vector=len(vector_hits),
            input_bm25=len(bm25_hits),
            merged=len(result),
        )

        return result[:top_n]
