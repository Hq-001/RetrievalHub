"""解析器协议 - 文档解析为结构化内容。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from retrievalhub.core.models import DocFormat


@runtime_checkable
class DocumentParser(Protocol):
    """文档解析器接口。

    将原始文件内容解析为结构化的中间表示（AST / 扁平化键值）。
    解析器不负责分块，仅负责结构提取。
    """

    def parse(self, content: str, filename: str) -> Any:
        """解析文件内容，返回结构化中间表示。

        Args:
            content: 文件文本内容（已 UTF-8 解码）
            filename: 原始文件名（用于推断格式）

        Returns:
            结构化中间表示（AST 节点列表 / 扁平化键值对列表）
        """
        ...

    def supported_formats(self) -> list[DocFormat]:
        """返回此解析器支持的格式列表。"""
        ...
