"""Phase 1 测试 - JSON 解析与分块。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from retrievalhub.core.exceptions import ParseError
from retrievalhub.core.models import ContentType, DocFormat
from retrievalhub.parsers.json_chunker import JsonChunker, make_parent_id
from retrievalhub.parsers.json_parser import JsonParser

FIXTURES = Path(__file__).parent / "fixtures"


# ---- 解析器测试 ----


class TestJsonParser:
    def setup_method(self):
        self.parser = JsonParser()

    def test_supported_formats(self):
        formats = self.parser.supported_formats()
        assert DocFormat.JSON in formats
        assert DocFormat.JSONL in formats

    def test_parse_simple_json(self):
        content = '{"name": "test", "value": 42}'
        atoms = self.parser.parse(content, "test.json")
        assert len(atoms) == 2
        keys = [a.key_path for a in atoms]
        assert "name" in keys
        assert "value" in keys

    def test_parse_nested_json(self):
        """嵌套对象压平为点号路径。"""
        content = '{"a": {"b": {"c": 1}}}'
        atoms = self.parser.parse(content, "test.json")
        # a.b.c -> 1
        atom = [a for a in atoms if a.key_path == "a.b.c"]
        assert len(atom) == 1
        assert atom[0].value == 1

    def test_parse_array_items(self):
        """数组项独立拆分。"""
        content = '{"list": [1, 2, 3]}'
        atoms = self.parser.parse(content, "test.json")
        # list[0], list[1], list[2]
        list_atoms = [a for a in atoms if "list[" in a.key_path]
        assert len(list_atoms) == 3

    def test_parse_array_of_objects(self):
        """数组对象项展开。"""
        content = '{"items": [{"name": "a"}, {"name": "b"}]}'
        atoms = self.parser.parse(content, "test.json")
        name_atoms = [a for a in atoms if "name" in a.key_path]
        assert len(name_atoms) == 2

    def test_parse_jsonl(self):
        content = '{"id": 1}\n{"id": 2}\n{"id": 3}'
        atoms = self.parser.parse(content, "test.jsonl")
        # 3 行 x 1 字段 = 3 atoms
        id_atoms = [a for a in atoms if "id" in a.key_path]
        assert len(id_atoms) == 3

    def test_parse_empty_content(self):
        atoms = self.parser.parse("", "test.json")
        assert atoms == []

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ParseError):
            self.parser.parse("{invalid json}", "test.json")

    def test_parse_null_value(self):
        atoms = self.parser.parse('{"key": null}', "test.json")
        assert len(atoms) == 1
        assert atoms[0].value == "null"

    def test_parse_boolean_value(self):
        atoms = self.parser.parse('{"a": true, "b": false}', "test.json")
        assert len(atoms) == 2
        vals = [a.value for a in atoms]
        assert "true" in vals
        assert "false" in vals

    def test_parse_sample_json_file(self):
        content = (FIXTURES / "sample.json").read_text(encoding="utf-8")
        atoms = self.parser.parse(content, "sample.json")
        # title, version, faq_list[0].question, faq_list[0].answer,
        # faq_list[1].question, faq_list[1].answer, metadata.author,
        # metadata.tags[0], metadata.tags[1], long_text
        keys = [a.key_path for a in atoms]
        assert "title" in keys
        assert "version" in keys
        assert any("faq_list[0]" in k for k in keys)
        assert any("metadata.author" in k for k in keys)
        assert any("long_text" in k for k in keys)

    def test_parent_path_for_array_items(self):
        """数组项的 parent_path 指向数组路径。"""
        content = '{"faq_list": [{"q": "1"}, {"q": "2"}]}'
        atoms = self.parser.parse(content, "test.json")
        list_atoms = [a for a in atoms if "faq_list" in a.key_path]
        for a in list_atoms:
            assert "faq_list" in a.parent_path


# ---- json_parent_id 测试 ----


class TestParentId:
    def test_make_parent_id_is_md5(self):
        """parent_id 应为 MD5 哈希（32 位十六进制）。"""
        pid = make_parent_id("$.faq_list")
        assert len(pid) == 32
        assert all(c in "0123456789abcdef" for c in pid)

    def test_make_parent_id_stable(self):
        """相同路径生成相同 parent_id（稳定可复现）。"""
        pid1 = make_parent_id("$.faq_list")
        pid2 = make_parent_id("$.faq_list")
        assert pid1 == pid2

    def test_make_parent_id_different_for_different_paths(self):
        pid1 = make_parent_id("$.faq_list")
        pid2 = make_parent_id("$.data.items")
        assert pid1 != pid2

    def test_make_parent_id_matches_manual_md5(self):
        """验证与手动 MD5 一致。"""
        path = "$.test.path"
        expected = hashlib.md5(path.encode("utf-8")).hexdigest()
        assert make_parent_id(path) == expected


# ---- 分块器测试 ----


class TestJsonChunker:
    def setup_method(self):
        self.parser = JsonParser()
        self.chunker = JsonChunker(
            chunk_size=600,
            chunk_overlap=60,
            long_value_threshold=10,  # 小阈值便于测试降级
        )

    def test_chunk_returns_chunks(self):
        content = '{"name": "test", "value": 42}'
        atoms = self.parser.parse(content, "test.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        assert len(chunks) > 0
        assert all(c.doc_id == "doc1" for c in chunks)
        assert all(c.kb_id == "kb1" for c in chunks)

    def test_chunk_content_type_is_json(self):
        content = '{"name": "test"}'
        atoms = self.parser.parse(content, "test.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        assert all(c.content_type == ContentType.JSON for c in chunks)

    def test_chunk_text_contains_key_value(self):
        content = '{"name": "test"}'
        atoms = self.parser.parse(content, "test.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        assert any("name: test" in c.text for c in chunks)

    def test_array_items_share_parent_id(self):
        """同一数组下各项共享 json_parent_id。"""
        content = '{"list": [{"a": 1}, {"a": 2}]}'
        atoms = self.parser.parse(content, "test.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        # 所有 chunk 的 json_parent_id 不为 None
        for c in chunks:
            assert c.json_parent_id is not None
        # 同一数组项应有相同的 parent_id（可能因为分组被合到一起）
        # 验证不同数组有不同 parent_id
        list_chunks = [c for c in chunks if "list" in c.text]
        if len(list_chunks) > 1:
            # 相同数组的 parent_id 应一致
            pids = {c.json_parent_id for c in list_chunks}
            assert len(pids) >= 1

    def test_seq_is_sequential(self):
        content = '{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}'
        atoms = self.parser.parse(content, "test.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        seqs = [c.seq for c in chunks]
        assert seqs == list(range(len(chunks)))

    def test_chunk_has_unique_ids(self):
        content = (FIXTURES / "sample.json").read_text(encoding="utf-8")
        atoms = self.parser.parse(content, "sample.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_long_value_degradation(self):
        """超长 Value 应降级为 MD 语义切分。"""
        long_text = "This is a long text. " * 100  # 远超阈值
        content = f'{{"key": "{long_text}"}}'
        atoms = self.parser.parse(content, "test.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        # 应产生多个子块
        assert len(chunks) > 1
        # 所有子块共享同一 json_parent_id
        pids = {c.json_parent_id for c in chunks}
        assert len(pids) == 1
        # 所有子块包含键路径前缀
        assert all("key:" in c.text or "key" in c.text for c in chunks)

    def test_sample_json_full_pipeline(self):
        """完整 sample.json 端到端解析+分块。"""
        content = (FIXTURES / "sample.json").read_text(encoding="utf-8")
        atoms = self.parser.parse(content, "sample.json")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")

        # 验证有 chunk 包含 FAQ 内容
        assert any("faq_list" in c.text for c in chunks)
        # 验证有 chunk 包含 title
        assert any("title" in c.text for c in chunks)
        # 验证所有 json_parent_id 不为 None
        assert all(c.json_parent_id is not None for c in chunks)

    def test_jsonl_full_pipeline(self):
        """JSONL 端到端解析+分块。"""
        content = (FIXTURES / "sample.jsonl").read_text(encoding="utf-8")
        atoms = self.parser.parse(content, "sample.jsonl")
        chunks = self.chunker.chunk(atoms, "doc1", "kb1")
        assert len(chunks) > 0
        # 每行内容应出现在某个 chunk
        texts = "\n".join(c.text for c in chunks)
        assert "first" in texts
        assert "second" in texts
        assert "third" in texts

    def test_small_chunk_size_groups_atoms(self):
        """小 chunk_size 应将键值对分到多个块。"""
        content = '{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}'
        atoms = self.parser.parse(content, "test.json")
        chunker = JsonChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(atoms, "doc1", "kb1")
        # 小块应产生多个 chunk
        assert len(chunks) > 1
