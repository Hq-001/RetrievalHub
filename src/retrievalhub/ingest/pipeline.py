"""入库流水线 - 解析 -> 分块 -> 批量嵌入 -> 写存储。

编排异步入库的完整流程，更新文档状态。
"""

from __future__ import annotations

from pathlib import Path

from retrievalhub.core.exceptions import IngestError
from retrievalhub.core.models import Chunk, DocFormat, DocStatus
from retrievalhub.embedders.embedder import MockEmbedder
from retrievalhub.ingest.file_handler import (
    compute_content_hash,
    detect_format,
    read_file_content,
    validate_content,
)
from retrievalhub.parsers.json_chunker import JsonChunker
from retrievalhub.parsers.json_parser import JsonParser
from retrievalhub.parsers.md_chunker import MarkdownChunker
from retrievalhub.parsers.md_parser import MarkdownParser
from retrievalhub.storage.lancedb_store import LanceDBStorage
from retrievalhub.storage.metadata_store import MetadataStore
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class IngestPipeline:
    """入库流水线编排器。

    异步执行：解析 -> 分块 -> 批量嵌入 -> 写 LanceDB + 元数据库。
    """

    def __init__(
        self,
        storage: LanceDBStorage,
        metadata: MetadataStore,
        embedder: MockEmbedder | None = None,
        md_chunk_size: int = 800,
        md_chunk_overlap: int = 100,
        max_section_depth: int = 3,
        json_chunk_size: int = 600,
        json_chunk_overlap: int = 60,
        json_long_value_threshold: int = 2000,
    ) -> None:
        self._storage = storage
        self._metadata = metadata
        self._embedder = embedder or MockEmbedder()

        # 解析器与分块器
        self._md_parser = MarkdownParser()
        self._md_chunker = MarkdownChunker(
            chunk_size=md_chunk_size,
            chunk_overlap=md_chunk_overlap,
            max_section_depth=max_section_depth,
        )
        self._json_parser = JsonParser()
        self._json_chunker = JsonChunker(
            chunk_size=json_chunk_size,
            chunk_overlap=json_chunk_overlap,
            long_value_threshold=json_long_value_threshold,
        )

    async def ingest(
        self,
        doc_id: str,
        file_path: str | Path,
        kb_id: str,
        content_hash: str,
        collection: str,
    ) -> None:
        """执行完整入库流水线。

        Args:
            doc_id: 文档 ID
            file_path: 文件路径
            kb_id: 知识库 ID
            content_hash: 内容哈希
            collection: LanceDB 集合名
        """
        logger.info("ingest_start", doc_id=doc_id, kb_id=kb_id)

        try:
            # 1. 读取文件
            content = read_file_content(Path(file_path))
            fmt = detect_format(str(file_path))

            # 2. 校验
            validate_content(content, fmt)

            # 3. 解析 + 分块
            chunks = self._parse_and_chunk(content, fmt, doc_id, kb_id)
            logger.info(
                "ingest_chunked",
                doc_id=doc_id,
                chunk_count=len(chunks),
                format=fmt.value,
            )

            if not chunks:
                raise IngestError("解析后无有效分块")

            # 4. 批量嵌入
            texts = [c.text for c in chunks]
            vectors = await self._embedder.embed(texts)

            # 5. 写存储（向量 + 全文索引）
            await self._storage.upsert(collection, chunks, vectors)

            # 6. 更新元数据状态
            self._metadata.update_document_status(
                doc_id, DocStatus.READY, chunk_count=len(chunks)
            )

            logger.info(
                "ingest_complete",
                doc_id=doc_id,
                chunk_count=len(chunks),
            )

        except Exception as e:
            logger.error("ingest_failed", doc_id=doc_id, error=str(e))
            self._metadata.update_document_status(
                doc_id, DocStatus.FAILED, error_message=str(e)
            )
            raise

    def _parse_and_chunk(
        self, content: str, fmt: DocFormat, doc_id: str, kb_id: str
    ) -> list[Chunk]:
        """按格式路由解析与分块。"""
        if fmt == DocFormat.MD:
            parsed = self._md_parser.parse(content, filename=f"doc.{fmt.value}")
            return self._md_chunker.chunk(parsed, doc_id, kb_id)

        elif fmt in (DocFormat.JSON, DocFormat.JSONL):
            filename = f"doc.{fmt.value}"
            parsed = self._json_parser.parse(content, filename=filename)
            return self._json_chunker.chunk(parsed, doc_id, kb_id)

        else:
            raise IngestError(f"不支持的格式: {fmt}")
