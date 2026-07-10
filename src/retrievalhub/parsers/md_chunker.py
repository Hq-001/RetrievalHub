"""Markdown 分块器 - 递归标题继承法。

核心策略：
- H1/H2 作为强制切分边界（遇 H1/H2 开新块）
- 每个块继承标题路径生成 section_path（深度 <= 3）
- H3+ 不强制切分，但 section_path 最多继承至 H3
- 代码块强制完整，超长独立成块
- 嵌套列表继承深度 <= 3 级压平
"""

from __future__ import annotations

import uuid

from retrievalhub.core.models import Chunk, ContentType
from retrievalhub.parsers.md_parser import MDNode, MarkdownParser


class MarkdownChunker:
    """Markdown 递归标题继承法分块器。

    配置参数（通过构造函数注入，默认值来自 .env）：
        chunk_size: 块大小上限（字符），默认 800
        chunk_overlap: 重叠量（字符），默认 100
        max_section_depth: 标题继承深度上限，默认 3
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        max_section_depth: int = 3,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_section_depth = max_section_depth

    def chunk(
        self,
        parsed: list[MDNode],
        doc_id: str,
        kb_id: str,
    ) -> list[Chunk]:
        """将 AST 节点列表切分为 Chunk。

        Args:
            parsed: MarkdownParser.parse() 返回的 MDNode 列表
            doc_id: 所属文档 ID
            kb_id: 所属知识库 ID

        Returns:
            Chunk 列表，含 section_path 和 code_language 元数据
        """
        chunks: list[Chunk] = []
        current_text = ""
        current_offset = 0
        section_stack: list[tuple[int, str]] = []  # [(level, title), ...]
        seq = 0

        for node in parsed:
            if node.node_type == "heading":
                # 遇到 H1/H2 -> 强制切分当前块
                if current_text.strip():
                    chunks.extend(
                        self._finalize_chunk(
                            current_text, doc_id, kb_id, seq,
                            section_stack, current_offset,
                        )
                    )
                    seq += len(chunks) - seq  # 保持 seq 连续
                    current_text = ""

                # 更新标题栈
                self._update_section_stack(section_stack, node.level, node.content)

            elif node.node_type == "code_block":
                # 代码块强制完整：先闭合当前文本块
                if current_text.strip():
                    chunks.extend(
                        self._finalize_chunk(
                            current_text, doc_id, kb_id, seq,
                            section_stack, current_offset,
                        )
                    )
                    seq = len(chunks)
                    current_text = ""

                # 代码块独立成块
                section_path = self._build_section_path(section_stack)
                code_text = node.content.rstrip("\n")
                if code_text:
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            kb_id=kb_id,
                            seq=seq,
                            text=code_text,
                            content_type=ContentType.MD,
                            section_path=section_path,
                            code_language=node.code_language or None,
                            char_offset=current_offset,
                        )
                    )
                    seq += 1
                    current_offset += len(code_text)

            elif node.node_type in ("paragraph", "text", "list"):
                # 累积文本，超限时切分
                node_text = node.content
                if current_text and len(current_text) + len(node_text) + 1 > self.chunk_size:
                    # 当前块已满，先输出
                    finalized = self._finalize_chunk(
                        current_text, doc_id, kb_id, seq,
                        section_stack, current_offset,
                    )
                    chunks.extend(finalized)
                    seq = len(chunks)
                    # 保留 overlap
                    if self.chunk_overlap > 0 and len(current_text) > self.chunk_overlap:
                        current_text = current_text[-self.chunk_overlap:] + "\n" + node_text
                    else:
                        current_text = node_text
                    current_offset += len(current_text)
                else:
                    if current_text:
                        current_text += "\n" + node_text
                    else:
                        current_text = node_text

        # 最后一个块
        if current_text.strip():
            chunks.extend(
                self._finalize_chunk(
                    current_text, doc_id, kb_id, seq,
                    section_stack, current_offset,
                )
            )

        # 重新编号 seq
        for i, c in enumerate(chunks):
            c.seq = i

        return chunks

    def _update_section_stack(
        self, stack: list[tuple[int, str]], level: int, title: str
    ) -> None:
        """更新标题栈，维护深度 <= max_section_depth。"""
        # 弹出所有 >= 当前 level 的标题
        while stack and stack[-1][0] >= level:
            stack.pop()
        # 仅保留到 max_section_depth 层级
        if level <= self.max_section_depth:
            stack.append((level, title))

    def _build_section_path(self, stack: list[tuple[int, str]]) -> str:
        """从标题栈构建 section_path，如 '第三章 > 注意力机制'。"""
        return " > ".join(title for _, title in stack)

    def _finalize_chunk(
        self,
        text: str,
        doc_id: str,
        kb_id: str,
        seq: int,
        section_stack: list[tuple[int, str]],
        offset: int,
    ) -> list[Chunk]:
        """将文本切分为最终 Chunk（处理超长文本二次切分）。

        Returns:
            Chunk 列表（通常 1 个，超长时多个）
        """
        text = text.strip()
        if not text:
            return []

        section_path = self._build_section_path(section_stack)

        # 超长文本二次切分（带 overlap）
        if len(text) > self.chunk_size:
            return self._split_long_text(
                text, doc_id, kb_id, seq, section_path, offset
            )

        return [
            Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                kb_id=kb_id,
                seq=seq,
                text=text,
                content_type=ContentType.MD,
                section_path=section_path,
                char_offset=offset,
            )
        ]

    def _split_long_text(
        self,
        text: str,
        doc_id: str,
        kb_id: str,
        base_seq: int,
        section_path: str,
        offset: int,
    ) -> list[Chunk]:
        """将超长文本按 chunk_size 切分（带 overlap）。"""
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        kb_id=kb_id,
                        seq=base_seq + idx,
                        text=chunk_text,
                        content_type=ContentType.MD,
                        section_path=section_path,
                        char_offset=offset + start,
                    )
                )
                idx += 1

            if end >= text_len:
                break
            # overlap 回退
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks
