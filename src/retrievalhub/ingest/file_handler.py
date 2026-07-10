"""文件处理工具 - 格式校验、内容哈希、原文存储。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from retrievalhub.core.exceptions import ParseError, UnsupportedFormatError
from retrievalhub.core.models import DocFormat

# 支持的扩展名 -> DocFormat 映射
EXT_FORMAT_MAP: dict[str, DocFormat] = {
    ".md": DocFormat.MD,
    ".markdown": DocFormat.MD,
    ".json": DocFormat.JSON,
    ".jsonl": DocFormat.JSONL,
}


def detect_format(filename: str) -> DocFormat:
    """通过文件扩展名推断格式。

    Args:
        filename: 文件名

    Returns:
        DocFormat 枚举值

    Raises:
        UnsupportedFormatError: 不支持的扩展名
    """
    ext = Path(filename).suffix.lower()
    if ext not in EXT_FORMAT_MAP:
        raise UnsupportedFormatError(ext)
    return EXT_FORMAT_MAP[ext]


def validate_content(content: str, fmt: DocFormat) -> None:
    """内容嗅探校验。

    Args:
        content: 文件文本内容
        fmt: 文档格式

    Raises:
        ParseError: 内容不合法
    """
    if not content.strip():
        raise ParseError("文件内容为空")

    if fmt in (DocFormat.JSON, DocFormat.JSONL):
        import orjson

        if fmt == DocFormat.JSONL:
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if not lines:
                raise ParseError("JSONL 文件无有效行")
            for i, line in enumerate(lines, 1):
                try:
                    orjson.loads(line)
                except Exception as e:
                    raise ParseError(f"JSONL 第 {i} 行解析失败: {e}") from e
        else:
            try:
                orjson.loads(content)
            except Exception as e:
                raise ParseError(f"JSON 解析失败: {e}") from e


def compute_content_hash(content: str) -> str:
    """计算内容的 SHA-256 哈希（用于去重）。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def save_original_file(content: str, filename: str, storage_dir: Path) -> Path:
    """保存原始文件到存储目录。

    Args:
        content: 文件内容
        filename: 原始文件名
        storage_dir: 存储根目录

    Returns:
        保存的文件路径
    """
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path


def read_file_content(file_path: Path) -> str:
    """读取文件内容（UTF-8 解码）。"""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ParseError(f"文件非 UTF-8 编码: {e}") from e
