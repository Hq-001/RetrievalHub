"""Markdown 解析器 - AST 提取标题层级、代码块、段落、列表。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from markdown_it import MarkdownIt

from retrievalhub.core.models import DocFormat


@dataclass
class MDNode:
    """Markdown AST 节点 - 解析后的中间表示。"""

    node_type: str  # heading | code_block | paragraph | list | text
    content: str = ""
    level: int = 0  # heading: 1-6, 其他: 0
    code_language: str = ""  # 仅 code_block
    children: list[Any] = field(default_factory=list)


class MarkdownParser:
    """Markdown 解析器，将 MD 解析为 AST 节点列表。

    使用 markdown-it-py 解析为 AST（非正则），提取：
    - 标题层级（H1~H6）
    - 代码块（含语言标识 code_language）
    - 段落
    - 列表（含嵌套）
    """

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark", {"html": False})

    def parse(self, content: str, filename: str = "") -> list[MDNode]:
        """解析 Markdown 文本为 AST 节点列表。

        Args:
            content: Markdown 文本（已 UTF-8 解码）
            filename: 原始文件名

        Returns:
            MDNode 列表，保留文档顺序
        """
        tokens = self._md.parse(content, {})
        nodes: list[MDNode] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            node = self._token_to_node(tok, tokens, i)
            if node is not None:
                nodes.append(node)
            i = self._next_token_index(tok, tokens, i)
        return nodes

    def _token_to_node(
        self, tok: Any, tokens: list[Any], idx: int
    ) -> MDNode | None:
        """将 markdown-it token 转为 MDNode。"""
        ttype = tok.type

        if ttype == "heading_open":
            level = int(tok.tag[1])  # h1 -> 1, h2 -> 2
            inline_tok = tokens[idx + 1] if idx + 1 < len(tokens) else None
            text = ""
            if inline_tok and inline_tok.type == "inline":
                text = inline_tok.content
            return MDNode(node_type="heading", content=text, level=level)

        elif ttype == "fence" or ttype == "code_block":
            lang = tok.info.strip() if tok.info else ""
            return MDNode(
                node_type="code_block",
                content=tok.content,
                code_language=lang,
            )

        elif ttype == "paragraph_open":
            inline_tok = tokens[idx + 1] if idx + 1 < len(tokens) else None
            text = ""
            if inline_tok and inline_tok.type == "inline":
                text = inline_tok.content
            return MDNode(node_type="paragraph", content=text)

        elif ttype == "bullet_list_open" or ttype == "ordered_list_open":
            items = self._collect_list_items(tokens, idx)
            return MDNode(node_type="list", content=items)

        elif ttype == "inline":
            if tok.content.strip():
                return MDNode(node_type="text", content=tok.content)
            return None

        return None

    def _collect_list_items(self, tokens: list[Any], start: int) -> str:
        """收集列表项内容，压平为文本。"""
        items: list[str] = []
        depth = 0
        i = start + 1
        while i < len(tokens):
            tok = tokens[i]
            if tok.type in ("bullet_list_open", "ordered_list_open"):
                depth += 1
            elif tok.type in ("bullet_list_close", "ordered_list_close"):
                if depth == 0:
                    break
                depth -= 1
            elif tok.type == "inline" and tok.content.strip():
                items.append(tok.content.strip())
            i += 1
        return "\n".join(f"- {item}" for item in items)

    def _next_token_index(self, tok: Any, tokens: list[Any], idx: int) -> int:
        """跳过当前 token 及其配对的 close token 内部内容。

        对于 _open/_close 对，跳到对应的 _close 之后。
        对于自闭合 token，跳到下一个。
        """
        ttype = tok.type
        if ttype.endswith("_open"):
            close_type = ttype.replace("_open", "_close")
            depth = 1
            i = idx + 1
            while i < len(tokens) and depth > 0:
                if tokens[i].type == ttype:
                    depth += 1
                elif tokens[i].type == close_type:
                    depth -= 1
                i += 1
            return i
        return idx + 1

    def supported_formats(self) -> list[DocFormat]:
        return [DocFormat.MD]
