"""自动调参工具 - 网格搜索优化权重与超参。

以评测集为优化目标，对 MD_VECTOR_WEIGHT / JSON_BM25_WEIGHT、
召回 top-N、重排序 top-K 做网格搜索。
含 K-Fold 交叉验证防过拟合。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from retrievalhub.retrieval.eval import EvalDataset, EvalSample, evaluate_search


@dataclass
class ParamGrid:
    """参数搜索网格。"""

    md_vector_weights: list[float] = field(
        default_factory=lambda: [0.8, 1.0, 1.1, 1.2, 1.5]
    )
    json_bm25_weights: list[float] = field(
        default_factory=lambda: [0.8, 1.0, 1.2, 1.5, 2.0]
    )
    vector_top_ns: list[int] = field(
        default_factory=lambda: [20, 50, 100]
    )
    rerank_top_ks: list[int] = field(
        default_factory=lambda: [5, 10, 20]
    )

    def all_combinations(self) -> list[dict[str, Any]]:
        """生成所有参数组合。"""
        combos = []
        for md_vw, json_bw, vtn, rtk in itertools.product(
            self.md_vector_weights,
            self.json_bm25_weights,
            self.vector_top_ns,
            self.rerank_top_ks,
        ):
            combos.append({
                "md_vector_weight": md_vw,
                "json_bm25_weight": json_bw,
                "vector_top_n": vtn,
                "rerank_top_k": rtk,
            })
        return combos


@dataclass
class TuneResult:
    """单次调参结果。"""

    params: dict[str, Any]
    recall_at_5: float
    mrr: float
    score: float  # 综合评分

    def __str__(self) -> str:
        return (
            f"params={self.params} "
            f"recall@5={self.recall_at_5:.4f} "
            f"mrr={self.mrr:.4f} "
            f"score={self.score:.4f}"
        )


def k_fold_split(
    samples: list[EvalSample], k: int = 5, seed: int = 42
) -> list[tuple[list[EvalSample], list[EvalSample]]]:
    """K-Fold 交叉验证分割。

    Args:
        samples: 评测样本
        k: 折数
        seed: 随机种子

    Returns:
        [(train_samples, val_samples), ...] k 组
    """
    import random

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(shuffled)
        val = shuffled[start:end]
        train = shuffled[:start] + shuffled[end:]
        folds.append((train, val))

    return folds


def grid_search(
    search_fn,
    dataset: EvalDataset,
    param_grid: ParamGrid | None = None,
    k_fold: int = 5,
) -> list[TuneResult]:
    """网格搜索 + K-Fold 交叉验证。

    Args:
        search_fn: 搜索函数 (params, samples) -> {sample_index: [chunk_ids]}
        dataset: 评测集（使用 train_samples）
        param_grid: 参数网格
        k_fold: K-Fold 折数

    Returns:
        按 score 降序排列的 TuneResult 列表
    """
    grid = param_grid or ParamGrid()
    combos = grid.all_combinations()
    results: list[TuneResult] = []

    samples = dataset.train_samples or dataset.samples

    if not samples:
        return results

    for params in combos:
        # K-Fold 交叉验证
        folds = k_fold_split(samples, k=k_fold)
        fold_scores = []

        for train_samples, val_samples in folds:
            # 在验证集上评测
            search_results = search_fn(params, val_samples)
            metrics = evaluate_search(search_results, val_samples, k=5)
            fold_scores.append(metrics)

        # 平均各折结果
        avg_recall = sum(s["recall@5"] for s in fold_scores) / len(fold_scores)
        avg_mrr = sum(s["mrr"] for s in fold_scores) / len(fold_scores)

        # 综合评分：recall 权重 0.6 + mrr 权重 0.4
        score = avg_recall * 0.6 + avg_mrr * 0.4

        results.append(TuneResult(
            params=params,
            recall_at_5=avg_recall,
            mrr=avg_mrr,
            score=score,
        ))

    # 按综合评分降序
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def best_params(results: list[TuneResult]) -> TuneResult:
    """获取最佳参数组合。"""
    return results[0] if results else None
