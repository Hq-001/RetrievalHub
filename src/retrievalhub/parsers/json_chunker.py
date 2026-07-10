"""JSON 分块器 - 扁平化原子键值对 -> 检索 Chunk。

核心策略：
- 每个原子键值对成为可检索文本
- 数组项共享 json_parent_id（父路径的 MD5 哈希，稳定可复现）
- 语义相关键值对分组（不超过 chunk_size 上限）
- 超长 Value (>threshold tokens) 降级为 MD 语义切分
"""

from __future__ import annotations

import hashlib
import uuid

from retrievalhub.core.models import Chunk, ContentType
from retrievalhub.parsers.json_parser import JSONAtom
from retrievalhub.parsers.md_chunker import MarkdownChunker
from retrievalhub.parsers.md_parser import MarkdownParser


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（约 4 字符 = 1 token）。"""
    return max(1, len(text) // 4)


def make_parent_id(parent_path: str) -> str:
    """生成 json_parent_id - 父路径的 MD5 哈希。

    以路径而非自增序号为基准，确保同一数组下所有子块拥有
    稳定且一致的 parent_id（与文档内容无关、可跨次入库复现）。

    Args:
        parent_path: JSON Path，如 "$.faq_list" 或 "$.data.items"

    Returns:
        32 位 MD5 哈希字符串
    """
    return hashlib.md5(parent_path.encode("utf-8")).hexdigest()


class JsonChunker:
    """JSON 分块器。

    配置参数：
        chunk_size: 块大小上限（字符），默认 600
        chunk_overlap: 重叠量（字符），默认 60
        long_value_threshold: 超长 Value 阈值（tokens），默认 2000
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 60,
        long_value_threshold: int = 2000,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.long_value_threshold = long_value_threshold
        # 降级用 MD 分块器
        self._md_parser = MarkdownParser()
        self._md_chunker = MarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        atoms: list[JSONAtom],
        doc_id: str,
        kb_id: str,
    ) -> list[Chunk]:
        """将扁平化原子键值对列表切分为 Chunk。

        Args:
            atoms: JsonParser.parse() 返回的 JSONAtom 列表
            doc_id: 所属文档 ID
            kb_id: 所属知识库 ID

        Returns:
            Chunk 列表，含 json_parent_id 元数据
        """
        chunks: list[Chunk] = []
        seq = 0
        current_text = ""

        for atom in atoms:
            # 格式化为可检索文本
            atom_text = self._format_atom(atom)

            # 超长 Value 降级为 MD 语义切分
            if isinstance(atom.value, str) and _estimate_tokens(atom.value) > self.long_value_threshold:
                # 先输出累积的文本
                if current_text.strip():
                    chunks.append(self._make_chunk(
                        current_text.strip(), doc_id, kb_id, seq, atom
                    ))
                    seq += 1
                    current_text = ""

                # 降级：将超长 Value 用 MD 分块器切分
                degraded = self._degrade_long_value(atom, doc_id, kb_id, seq)
                chunks.extend(degraded)
                seq += len(degraded)
                continue

            # 检查是否超出 chunk_size
            if current_text and len(current_text) + len(atom_text) + 1 > self.chunk_size:
                # 输出当前块
                chunks.append(self._make_chunk(
                    current_text.strip(), doc_id, kb_id, seq, atom
                ))
                seq += 1
                current_text = atom_text
            else:
                if current_text:
                    current_text += "\n" + atom_text
                else:
                    current_text = atom_text

        # 最后一个块
        if current_text.strip():
            # 找最后一个 atom 用于 metadata
            last_atom = atoms[-1] if atoms else JSONAtom(
                key_path="", value="", parent_path="$"
            )
            chunks.append(self._make_chunk(
                current_text.strip(), doc_id, kb_id, seq, last_atom
            ))

        # 重新编号 seq
        for i, c in enumerate(chunks):
            c.seq = i

        return chunks

    def _format_atom(self, atom: JSONAtom) -> str:
        """将原子键值对格式化为可检索文本。"""
        if isinstance(atom.value, str):
            return f"{atom.key_path}: {atom.value}"
        elif atom.value is None:
            return f"{atom.key_path}: null"
        elif isinstance(atom.value, bool):
            return f"{atom.key_path}: {'true' if atom.value else 'false'}"
        else:
            return f"{atom.key_path}: {atom.value}"

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        kb_id: str,
        seq: int,
        atom: JSONAtom,
    ) -> Chunk:
        """创建单个 JSON Chunk。"""
        return Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            kb_id=kb_id,
            seq=seq,
            text=text,
            content_type=ContentType.JSON,
            json_parent_id=make_parent_id(atom.parent_path),
            char_offset=0,
        )

    def _degrade_long_value(
        self, atom: JSONAtom, doc_id: str, kb_id: str, base_seq: int
    ) -> list[Chunk]:
        """超长 Value 降级为 MD 语义切分。

        将超长文本用 MarkdownChunker 二次切分，
        子块共享同一 json_parent_id 与键路径。
        """
        # 将超长 Value 当作 Markdown 文本解析分块
        md_nodes = self._md_parser.parse(atom.value)
        if not md_nodes:
            # 如果不是有效 MD，直接按字符切分
            value = atom.value if isinstance(atom.value, str) else str(atom.value)
            return self._fallback_split(value, atom, doc_id, kb_id, base_seq)

        md_chunks = self._md_chunker.chunk(md_nodes, doc_id, kb_id)

        # 转换为 JSON 类型的 Chunk，附加 json_parent_id
        result: list[Chunk] = []
        parent_id = make_parent_id(atom.parent_path)
        for i, mc in enumerate(md_chunks):
            # 在文本前加上键路径前缀
            prefixed_text = f"{atom.key_path}:\n{mc.text}"
            result.append(Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                kb_id=kb_id,
                seq=base_seq + i,
                text=prefixed_text,
                content_type=ContentType.JSON,
                json_parent_id=parent_id,
                char_offset=mc.char_offset,
            ))

        return result

    def _fallback_split(
        self, text: str, atom: JSONAtom, doc_id: str, kb_id: str, base_seq: int
    ) -> list[Chunk]:
        """无法用 MD 解析时的兜底字符切分。"""
        chunks: list[Chunk] = []
        parent_id = make_parent_id(atom.parent_path)
        start = 0
        idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                prefixed = f"{atom.key_path}: {chunk_text}"
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    kb_id=kb_id,
                    seq=base_seq + idx,
                    text=prefixed,
                    content_type=ContentType.JSON,
                    json_parent_id=parent_id,
                    char_offset=start,
                ))
                idx += 1
            if end >= text_len:
                break
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks
