"""JSON 解析器 - orjson 高性能解析 + 扁平化压平。

扁平化规则：
- 嵌套对象压平为原子键值对，键用点号路径表示
  {"a":{"b":1}} -> "a.b: 1"
- 数组项独立拆分为多个 Chunk，共享 json_parent_id
- JSONL 每行作为独立顶层对象处理
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import orjson

from retrievalhub.core.exceptions import ParseError
from retrievalhub.core.models import DocFormat


@dataclass
class JSONAtom:
    """JSON 扁平化后的原子键值对。"""

    key_path: str  # 点号路径，如 "a.b" 或 "faq_list[0].question"
    value: Any  # 值（标量或长文本）
    parent_path: str  # 父路径，如 "$.faq_list"，用于生成 json_parent_id
    is_array_item: bool = False  # 是否为数组项


class JsonParser:
    """JSON 解析器，将 JSON/JSONL 解析为扁平化原子键值对列表。"""

    def parse(self, content: str, filename: str = "") -> list[JSONAtom]:
        """解析 JSON/JSONL 文本为原子键值对列表。

        Args:
            content: JSON/JSONL 文本
            filename: 原始文件名（用于推断格式）

        Returns:
            JSONAtom 列表
        """
        content = content.strip()
        if not content:
            return []

        # 判断 JSONL 还是 JSON
        is_jsonl = filename.endswith(".jsonl") or self._is_jsonl(content)

        if is_jsonl:
            return self._parse_jsonl(content)
        else:
            return self._parse_json(content)

    def _parse_json(self, content: str) -> list[JSONAtom]:
        """解析单个 JSON 对象。"""
        try:
            data = orjson.loads(content)
        except orjson.JSONDecodeError as e:
            raise ParseError(f"JSON 解析失败: {e}") from e

        atoms: list[JSONAtom] = []
        self._flatten(data, "", "$", atoms)
        return atoms

    def _parse_jsonl(self, content: str) -> list[JSONAtom]:
        """解析 JSONL（每行一个独立 JSON 对象）。"""
        atoms: list[JSONAtom] = []
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ParseError(f"JSONL 第 {line_num} 行解析失败: {e}") from e

            # 每行作为独立顶层对象，路径前缀加行号
            line_prefix = f"$.line_{line_num}"
            self._flatten(data, "", line_prefix, atoms)

        return atoms

    def _flatten(
        self,
        obj: Any,
        current_key: str,
        parent_path: str,
        atoms: list[JSONAtom],
    ) -> None:
        """递归压平 JSON 对象为原子键值对。

        Args:
            obj: 当前处理的值
            current_key: 当前键路径（如 "a.b"）
            parent_path: 父路径（如 "$.a"）
            atoms: 输出列表
        """
        if obj is None:
            atoms.append(JSONAtom(
                key_path=current_key,
                value="null",
                parent_path=parent_path,
            ))
        elif isinstance(obj, bool):
            atoms.append(JSONAtom(
                key_path=current_key,
                value="true" if obj else "false",
                parent_path=parent_path,
            ))
        elif isinstance(obj, (int, float)):
            atoms.append(JSONAtom(
                key_path=current_key,
                value=obj,
                parent_path=parent_path,
            ))
        elif isinstance(obj, str):
            atoms.append(JSONAtom(
                key_path=current_key,
                value=obj,
                parent_path=parent_path,
            ))
        elif isinstance(obj, dict):
            new_parent = f"{parent_path}.{current_key}" if current_key else parent_path
            for k, v in obj.items():
                child_key = f"{current_key}.{k}" if current_key else k
                self._flatten(v, child_key, new_parent, atoms)
        elif isinstance(obj, list):
            # 数组项独立处理
            array_path = f"{parent_path}.{current_key}" if current_key else parent_path
            for idx, item in enumerate(obj):
                item_key = f"{current_key}[{idx}]" if current_key else f"[{idx}]"
                if isinstance(item, (dict, list)):
                    # 复杂数组项：递归展开
                    self._flatten(item, item_key, array_path, atoms)
                else:
                    # 简单数组项
                    atoms.append(JSONAtom(
                        key_path=item_key,
                        value=item,
                        parent_path=array_path,
                        is_array_item=True,
                    ))

    def _is_jsonl(self, content: str) -> bool:
        """判断是否为 JSONL 格式（多行 JSON）。"""
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if len(lines) <= 1:
            return False
        # 如果第一行不是以 { 或 [ 开头，不是 JSON
        try:
            orjson.loads(lines[0])
            return True
        except Exception:
            return False

    def supported_formats(self) -> list[DocFormat]:
        return [DocFormat.JSON, DocFormat.JSONL]
