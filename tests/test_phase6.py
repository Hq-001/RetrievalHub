"""Phase 6 测试 - LRU 缓存、评测集、自动调参。"""

from __future__ import annotations

import time

import pytest

from retrievalhub.retrieval.cache import LRUCache
from retrievalhub.retrieval.eval import (
    EvalDataset,
    EvalSample,
    create_default_eval_dataset,
    evaluate_search,
    mrr,
    recall_at_k,
)
from retrievalhub.retrieval.tune import (
    ParamGrid,
    TuneResult,
    grid_search,
    k_fold_split,
    best_params,
)


# ---- LRU 缓存测试 ----


class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUCache(maxsize=10, ttl_sec=60)
        cache.put("kb1", "query1", {"result": "hit"})
        result = cache.get("kb1", "query1")
        assert result is not None
        assert result["result"] == "hit"

    def test_miss(self):
        cache = LRUCache(maxsize=10, ttl_sec=60)
        result = cache.get("kb1", "nonexistent")
        assert result is None

    def test_lru_eviction(self):
        cache = LRUCache(maxsize=3, ttl_sec=60)
        cache.put("kb1", "q1", 1)
        cache.put("kb1", "q2", 2)
        cache.put("kb1", "q3", 3)
        # 访问 q1 使其变为最新
        cache.get("kb1", "q1")
        # 写入 q4，应淘汰最久未使用的 q2
        cache.put("kb1", "q4", 4)
        assert cache.get("kb1", "q1") is not None  # q1 保留
        assert cache.get("kb1", "q2") is None     # q2 被淘汰
        assert cache.get("kb1", "q3") is not None  # q3 保留
        assert cache.get("kb1", "q4") is not None  # q4 保留

    def test_ttl_expiry(self):
        cache = LRUCache(maxsize=10, ttl_sec=0)  # 立即过期
        cache.put("kb1", "q1", "data")
        time.sleep(0.1)
        result = cache.get("kb1", "q1")
        assert result is None

    def test_doc_versions_hash_eviction(self):
        """文档变更 -> doc_versions_hash 失配 -> 旧缓存失效。"""
        cache = LRUCache(maxsize=10, ttl_sec=60)
        cache.update_kb_version("kb1", "hash_v1")
        cache.put("kb1", "q1", "result1")

        # 模拟文档变更 -> 哈希更新
        cache.update_kb_version("kb1", "hash_v2")

        # 旧哈希的查询应失效
        result = cache.get("kb1", "q1")
        assert result is None

    def test_partial_eviction_different_kb(self):
        """一个 KB 的文档变更不影响其他 KB 的缓存。"""
        cache = LRUCache(maxsize=10, ttl_sec=60)
        cache.update_kb_version("kb1", "v1")
        cache.update_kb_version("kb2", "v1")
        cache.put("kb1", "q1", "result1")
        cache.put("kb2", "q1", "result2")

        # kb1 文档变更
        cache.update_kb_version("kb1", "v2")

        # kb1 失效，kb2 保留
        assert cache.get("kb1", "q1") is None
        assert cache.get("kb2", "q1") == "result2"

    def test_invalidate_kb(self):
        cache = LRUCache(maxsize=10, ttl_sec=60)
        cache.put("kb1", "q1", 1)
        cache.put("kb1", "q2", 2)
        count = cache.invalidate_kb("kb1")
        assert count == 2
        assert cache.get("kb1", "q1") is None
        assert cache.get("kb1", "q2") is None

    def test_stats(self):
        cache = LRUCache(maxsize=10, ttl_sec=60)
        cache.put("kb1", "q1", "data")
        cache.get("kb1", "q1")  # hit
        cache.get("kb1", "q2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert 0 < stats["hit_rate"] < 1

    def test_params_in_key(self):
        """不同 params 应有不同缓存键。"""
        cache = LRUCache(maxsize=10, ttl_sec=60)
        cache.put("kb1", "q1", "r1", params={"top_k": 5})
        cache.put("kb1", "q1", "r2", params={"top_k": 10})
        assert cache.get("kb1", "q1", params={"top_k": 5}) == "r1"
        assert cache.get("kb1", "q1", params={"top_k": 10}) == "r2"

    def test_compute_doc_versions_hash_stable(self):
        cache = LRUCache()
        h1 = cache.compute_doc_versions_hash("kb1", [("doc1", "2024-01-01"), ("doc2", "2024-01-02")])
        h2 = cache.compute_doc_versions_hash("kb1", [("doc1", "2024-01-01"), ("doc2", "2024-01-02")])
        assert h1 == h2

    def test_compute_doc_versions_hash_different(self):
        cache = LRUCache()
        h1 = cache.compute_doc_versions_hash("kb1", [("doc1", "2024-01-01")])
        h2 = cache.compute_doc_versions_hash("kb1", [("doc1", "2024-01-02")])
        assert h1 != h2


# ---- 评测集测试 ----


class TestEvalDataset:
    def test_create_dataset(self):
        ds = create_default_eval_dataset("kb1")
        assert len(ds.samples) == 10
        # 分层抽样覆盖各子类型
        sub_types = {s.sub_type for s in ds.samples}
        assert "text" in sub_types
        assert "code" in sub_types
        assert "list" in sub_types
        assert "kv" in sub_types
        assert "array" in sub_types
        assert "mixed" in sub_types

    def test_content_type_coverage(self):
        ds = create_default_eval_dataset("kb1")
        ct = {s.content_type for s in ds.samples}
        assert "md" in ct
        assert "json" in ct

    def test_split_train_and_blind(self):
        ds = create_default_eval_dataset("kb1")
        total = len(ds.samples)
        assert len(ds.train_samples) + len(ds.blind_samples) == total
        # 20% 盲测
        assert len(ds.blind_samples) == int(total * 0.2)

    def test_save_and_load(self, tmp_path):
        ds = create_default_eval_dataset("kb1")
        path = tmp_path / "eval.json"
        ds.save(path)
        loaded = EvalDataset.load(path)
        assert len(loaded.samples) == len(ds.samples)

    def test_recall_at_k_hit(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c3"]
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_partial(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c2", "c4"]
        # 2 个相关，命中 1 个
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_recall_at_k_no_hit(self):
        retrieved = ["c1", "c2"]
        relevant = ["c3"]
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_mrr_first_position(self):
        retrieved = ["c1", "c2"]
        relevant = ["c1"]
        assert mrr(retrieved, relevant) == 1.0

    def test_mrr_second_position(self):
        retrieved = ["c1", "c2"]
        relevant = ["c2"]
        assert mrr(retrieved, relevant) == 0.5

    def test_mrr_no_hit(self):
        retrieved = ["c1", "c2"]
        relevant = ["c3"]
        assert mrr(retrieved, relevant) == 0.0

    def test_evaluate_search(self):
        samples = [
            EvalSample(query="q1", relevant_chunk_ids=["c1"], content_type="md", sub_type="text"),
            EvalSample(query="q2", relevant_chunk_ids=["c3"], content_type="json", sub_type="kv"),
        ]
        results = {
            "0": ["c1", "c2"],  # hit at pos 1
            "1": ["c4", "c3"],  # hit at pos 2
        }
        metrics = evaluate_search(results, samples, k=5)
        assert "recall@5" in metrics
        assert "mrr" in metrics
        assert metrics["recall@5"] == 1.0  # both found in top 5
        assert abs(metrics["mrr"] - (1.0 + 0.5) / 2) < 0.001


# ---- 自动调参测试 ----


class TestAutoTune:
    def test_param_grid_combinations(self):
        grid = ParamGrid(
            md_vector_weights=[1.0, 1.1],
            json_bm25_weights=[1.0, 1.2],
            vector_top_ns=[50],
            rerank_top_ks=[10],
        )
        combos = grid.all_combinations()
        assert len(combos) == 4  # 2 x 2 x 1 x 1

    def test_k_fold_split(self):
        samples = [EvalSample(query=f"q{i}", relevant_chunk_ids=[], content_type="md", sub_type="text") for i in range(10)]
        folds = k_fold_split(samples, k=5)
        assert len(folds) == 5
        # 每折验证集 + 训练集 = 全集
        for train, val in folds:
            assert len(val) == 2
            assert len(train) == 8

    def test_grid_search_returns_sorted_results(self):
        """网格搜索应返回按 score 降序的结果。"""
        # Mock search function
        def mock_search(params, samples):
            results = {}
            for i, s in enumerate(samples):
                # 模拟返回结果
                results[str(i)] = s.relevant_chunk_ids if s.relevant_chunk_ids else ["fake"]
            return results

        ds = create_default_eval_dataset("kb1")
        grid = ParamGrid(
            md_vector_weights=[1.0, 1.1],
            json_bm25_weights=[1.0, 1.2],
            vector_top_ns=[50],
            rerank_top_ks=[10],
        )
        results = grid_search(mock_search, ds, param_grid=grid, k_fold=3)
        assert len(results) == 4
        # 验证降序
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_params(self):
        results = [
            TuneResult(params={"a": 1}, recall_at_5=0.9, mrr=0.8, score=0.86),
            TuneResult(params={"a": 2}, recall_at_5=0.7, mrr=0.6, score=0.66),
        ]
        best = best_params(results)
        assert best.params["a"] == 1
        assert best.score == 0.86

    def test_grid_search_empty_dataset(self):
        def mock_search(params, samples):
            return {}
        ds = EvalDataset(samples=[])
        results = grid_search(mock_search, ds, k_fold=3)
        assert results == []
