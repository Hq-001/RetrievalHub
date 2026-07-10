"""Phase 1 测试 - MD 解析与分块。"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrievalhub.core.models import ContentType, DocFormat
from retrievalhub.parsers.md_chunker import MarkdownChunker
from retrievalhub.parsers.md_parser import MarkdownParser

FIXTURES = Path(__file__).parent / "fixtures"


# ---- 解析器测试 ----


class TestMarkdownParser:
    def setup_method(self):
        self.parser = MarkdownParser()

    def test_supported_formats(self):
        assert DocFormat.MD in self.parser.supported_formats()

    def test_parse_simple_heading(self):
        nodes = self.parser.parse("# Title")
        headings = [n for n in nodes if n.node_type == "heading"]
        assert len(headings) == 1
        assert headings[0].level == 1
        assert headings[0].content == "Title"

    def test_parse_multi_level_headings(self):
        md = "# H1\n## H2\n### H3\n#### H4"
        nodes = self.parser.parse(md)
        headings = [n for n in nodes if n.node_type == "heading"]
        assert len(headings) == 4
        levels = [h.level for h in headings]
        assert levels == [1, 2, 3, 4]

    def test_parse_code_block_with_language(self):
        md = "```python\nprint('hello')\n```"
        nodes = self.parser.parse(md)
        code_blocks = [n for n in nodes if n.node_type == "code_block"]
        assert len(code_blocks) == 1
        assert code_blocks[0].code_language == "python"
        assert "print" in code_blocks[0].content

    def test_parse_code_block_without_language(self):
        md = "```\nplain code\n```"
        nodes = self.parser.parse(md)
        code_blocks = [n for n in nodes if n.node_type == "code_block"]
        assert len(code_blocks) == 1
        assert code_blocks[0].code_language == ""

    def test_parse_paragraph(self):
        nodes = self.parser.parse("This is a paragraph.")
        paras = [n for n in nodes if n.node_type == "paragraph"]
        assert len(paras) == 1
        assert "paragraph" in paras[0].content

    def test_parse_list(self):
        md = "- item1\n- item2\n- item3"
        nodes = self.parser.parse(md)
        lists = [n for n in nodes if n.node_type == "list"]
        assert len(lists) == 1
        assert "item1" in lists[0].content
        assert "item3" in lists[0].content

    def test_parse_sample_md_file(self):
        content = (FIXTURES / "sample.md").read_text(encoding="utf-8")
        nodes = self.parser.parse(content)
        headings = [n for n in nodes if n.node_type == "heading"]
        # 标题(H1), 第一章, 1.1, 1.2, 第二章, 2.1, 2.2, 2.2.1, 2.2.2, 2.2.2.1(H4), 第三章
        assert len(headings) == 11

    def test_parse_empty_content(self):
        nodes = self.parser.parse("")
        assert nodes == []


# ---- 分块器测试 ----


class TestMarkdownChunker:
    def setup_method(self):
        self.parser = MarkdownParser()
        self.chunker = MarkdownChunker(
            chunk_size=800,
            chunk_overlap=100,
            max_section_depth=3,
        )

    def test_chunk_returns_chunks(self):
        nodes = self.parser.parse("# Title\n\nSome content here.")
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        assert len(chunks) > 0
        assert all(c.doc_id == "doc1" for c in chunks)
        assert all(c.kb_id == "kb1" for c in chunks)

    def test_section_path_correct(self):
        md = "# Chapter 1\n## Section A\nContent of A."
        nodes = self.parser.parse(md)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        # 找到包含 "Content of A" 的 chunk
        content_chunks = [c for c in chunks if "Content of A" in c.text]
        assert len(content_chunks) >= 1
        assert "Chapter 1" in content_chunks[0].section_path
        assert "Section A" in content_chunks[0].section_path

    def test_section_path_depth_limit(self):
        """H4+ 内容应压平到 H3 层级，section_path 最多 H3。"""
        md = "# H1\n## H2\n### H3\n#### H4\nDeep content."
        nodes = self.parser.parse(md)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        deep_chunks = [c for c in chunks if "Deep content" in c.text]
        assert len(deep_chunks) >= 1
        sp = deep_chunks[0].section_path
        # section_path 不应包含 H4 标题
        assert "H4" not in sp
        # 应包含 H1 > H2 > H3
        assert "H1" in sp
        assert "H2" in sp
        assert "H3" in sp

    def test_h1_h2_force_split(self):
        """H1/H2 作为强制切分边界。"""
        md = "# Title A\nContent A\n# Title B\nContent B"
        nodes = self.parser.parse(md)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        # 应至少有 2 个块（Title A 和 Title B 分开）
        texts = [c.text for c in chunks]
        assert any("Content A" in t for t in texts)
        assert any("Content B" in t for t in texts)
        # Content A 和 Content B 不应在同一个块
        assert not any("Content A" in t and "Content B" in t for t in texts)

    def test_code_block_intact(self):
        """代码块强制完整，不在代码块内部切分。"""
        code = "def hello():\n    print('hello')\n    return 42"
        md = f"# Title\n\n```python\n{code}\n```\n\nAfter code."
        nodes = self.parser.parse(md)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        # 找到代码块 chunk
        code_chunks = [c for c in chunks if c.code_language == "python"]
        assert len(code_chunks) == 1
        assert "def hello" in code_chunks[0].text
        assert "return 42" in code_chunks[0].text
        # 代码块没有被拆分
        assert code_chunks[0].text.count("def hello") == 1

    def test_code_block_language_preserved(self):
        md = "# Title\n\n```javascript\nconst x = 10;\n```\n"
        nodes = self.parser.parse(md)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        js_chunks = [c for c in chunks if c.code_language == "javascript"]
        assert len(js_chunks) == 1

    def test_chunk_content_type(self):
        nodes = self.parser.parse("# Title\nContent.")
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        assert all(c.content_type == ContentType.MD for c in chunks)

    def test_seq_is_sequential(self):
        content = (FIXTURES / "sample.md").read_text(encoding="utf-8")
        nodes = self.parser.parse(content)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        seqs = [c.seq for c in chunks]
        assert seqs == list(range(len(chunks)))

    def test_chunk_has_unique_ids(self):
        content = (FIXTURES / "sample.md").read_text(encoding="utf-8")
        nodes = self.parser.parse(content)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # 无重复

    def test_long_text_split(self):
        """超长文本应被二次切分。"""
        long_text = "A" * 2000  # 超过 chunk_size
        nodes = self.parser.parse(f"# Title\n{long_text}")
        chunks = self.chunker.chunk(nodes, "doc1", "kb1", )
        # 使用小 chunk_size
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(nodes, "doc1", "kb1")
        assert len(chunks) > 1

    def test_sample_md_file_full_pipeline(self):
        """完整 sample.md 端到端解析+分块。"""
        content = (FIXTURES / "sample.md").read_text(encoding="utf-8")
        nodes = self.parser.parse(content)
        chunks = self.chunker.chunk(nodes, "doc1", "kb1")

        # 验证 section_path 包含正确标题
        section_paths = [c.section_path for c in chunks if c.section_path]
        assert any("第一章" in p for p in section_paths)
        assert any("第二章" in p for p in section_paths)
        assert any("第三章" in p for p in section_paths)

        # 验证代码块完整
        code_chunks = [c for c in chunks if c.code_language]
        assert any(c.code_language == "python" for c in code_chunks)
        assert any(c.code_language == "javascript" for c in code_chunks)

        # 验证 H4 内容压平到 H3
        cache_chunks = [c for c in chunks if "缓存层" in c.text or "压平" in c.text]
        if cache_chunks:
            assert "H4" not in cache_chunks[0].section_path
