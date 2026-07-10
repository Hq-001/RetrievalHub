"""评测集构建 - 按 content_type 分层抽样。

MD 子类型：纯文本 / 代码块 / 嵌套列表
JSON 子类型：键值对 / 数组对象 / 混合
预留 20% 盲测集 hold-out 防过拟合。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """单条评测样本。"""

    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    content_type: str  # md | json
    sub_type: str  # md: text|code|list, json: kv|array|mixed
    kb_id: str = ""


class EvalDataset(BaseModel):
    """评测集。"""

    samples: list[EvalSample] = Field(default_factory=list)
    train_samples: list[EvalSample] = Field(default_factory=list)
    blind_samples: list[EvalSample] = Field(default_factory=list)

    def split(self, blind_ratio: float = 0.2, seed: int = 42) -> None:
        """按比例分割训练集与盲测集。"""
        rng = random.Random(seed)
        shuffled = self.samples.copy()
        rng.shuffle(shuffled)
        n_blind = int(len(shuffled) * blind_ratio)
        self.blind_samples = shuffled[:n_blind]
        self.train_samples = shuffled[n_blind:]

    def save(self, path: Path) -> None:
        """保存评测集到 JSON 文件。"""
        data = {
            "samples": [s.model_dump() for s in self.samples],
            "train_samples": [s.model_dump() for s in self.train_samples],
            "blind_samples": [s.model_dump() for s in self.blind_samples],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        """从 JSON 文件加载评测集。"""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            samples=[EvalSample(**s) for s in data.get("samples", [])],
            train_samples=[EvalSample(**s) for s in data.get("train_samples", [])],
            blind_samples=[EvalSample(**s) for s in data.get("blind_samples", [])],
        )


# ---- 评测指标 ----


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@k: 前 k 个结果中命中的相关 chunk 比例。"""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR (Mean Reciprocal Rank): 第一个相关结果的倒数排名。"""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_search(
    search_results: list[dict[str, Any]],
    eval_samples: list[EvalSample],
    k: int = 5,
) -> dict[str, float]:
    """批量评测检索质量。

    Args:
        search_results: {sample_index: [chunk_id, ...]} 的映射
        eval_samples: 评测样本列表
        k: Recall@k 的 k 值

    Returns:
        {"recall@k": float, "mrr": float}
    """
    recalls = []
    mrrs = []

    for i, sample in enumerate(eval_samples):
        retrieved = search_results.get(str(i), [])
        recalls.append(recall_at_k(retrieved, sample.relevant_chunk_ids, k))
        mrrs.append(mrr(retrieved, sample.relevant_chunk_ids))

    return {
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
    }


# ---- 默认评测集 ----


def create_default_eval_dataset(kb_id: str = "eval-kb") -> EvalDataset:
    """创建默认评测集（分层抽样）。"""
    samples = [
        # MD - 纯文本
        EvalSample(
            query="概述",
            relevant_chunk_ids=[],
            content_type="md",
            sub_type="text",
            kb_id=kb_id,
        ),
        EvalSample(
            query="背景描述",
            relevant_chunk_ids=[],
            content_type="md",
            sub_type="text",
            kb_id=kb_id,
        ),
        # MD - 代码块
        EvalSample(
            query="python code",
            relevant_chunk_ids=[],
            content_type="md",
            sub_type="code",
            kb_id=kb_id,
        ),
        EvalSample(
            query="javascript",
            relevant_chunk_ids=[],
            content_type="md",
            sub_type="code",
            kb_id=kb_id,
        ),
        # MD - 嵌套列表
        EvalSample(
            query="item1",
            relevant_chunk_ids=[],
            content_type="md",
            sub_type="list",
            kb_id=kb_id,
        ),
        # JSON - 键值对
        EvalSample(
            query="title",
            relevant_chunk_ids=[],
            content_type="json",
            sub_type="kv",
            kb_id=kb_id,
        ),
        EvalSample(
            query="version",
            relevant_chunk_ids=[],
            content_type="json",
            sub_type="kv",
            kb_id=kb_id,
        ),
        # JSON - 数组对象
        EvalSample(
            query="faq",
            relevant_chunk_ids=[],
            content_type="json",
            sub_type="array",
            kb_id=kb_id,
        ),
        EvalSample(
            query="检索中间件",
            relevant_chunk_ids=[],
            content_type="json",
            sub_type="array",
            kb_id=kb_id,
        ),
        # JSON - 混合
        EvalSample(
            query="metadata author",
            relevant_chunk_ids=[],
            content_type="json",
            sub_type="mixed",
            kb_id=kb_id,
        ),
    ]

    dataset = EvalDataset(samples=samples)
    dataset.split(blind_ratio=0.2, seed=42)
    return dataset
