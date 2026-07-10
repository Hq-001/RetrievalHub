"""分块器协议 - 将解析后的结构化内容切分为检索 Chunk。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from retrievalhub.core.models import Chunk


@runtime_checkable
class Chunker(Protocol):
    """分块器接口。

    将解析器的输出（AST / 扁平化键值）切分为最小检索单元 Chunk。
    分块器负责生成 section_path、json_parent_id、code_language 等元数据。
    """

    def chunk(
        self,
        parsed: Any,
        doc_id: str,
        kb_id: str,
    ) -> list[Chunk]:
        """将解析结果切分为 Chunk 列表。

        Args:
            parsed: 解析器输出的结构化中间表示
            doc_id: 所属文档 ID
            kb_id: 所属知识库 ID

        Returns:
            Chunk 列表，已填充 section_path / json_parent_id / code_language 等元数据
        """
        ...
