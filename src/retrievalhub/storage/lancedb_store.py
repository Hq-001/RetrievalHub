"""LanceDB 存储实现 - 向量 + 全文(FTS) 混合检索。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.table import Table

from retrievalhub.core.exceptions import CollectionNotFoundError, StorageError
from retrievalhub.core.models import Chunk
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class LanceDBStorage:
    """LanceDB 存储实现。

    原生支持向量检索 + 全文检索(FTS/BM25)于同一存储，
    避免双索引一致性问题。
    """

    def __init__(self, uri: str = "./data/lancedb", distance_metric: str = "cosine") -> None:
        self._uri = uri
        self._distance_metric = distance_metric
        self._db: lancedb.DBConnection | None = None

    async def _ensure_db(self) -> lancedb.DBConnection:
        if self._db is None:
            await asyncio.to_thread(Path(self._uri).mkdir, parents=True, exist_ok=True)
            self._db = await asyncio.to_thread(lancedb.connect, self._uri)
        return self._db

    def _collection_name(self, kb_id: str) -> str:
        return f"kb_{kb_id}"

    async def create_collection(self, kb_id: str, dim: int) -> str:
        """创建知识库对应的集合，含向量列与全文索引。"""
        db = await self._ensure_db()
        collection = self._collection_name(kb_id)

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("kb_id", pa.string()),
            pa.field("seq", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("content_type", pa.string()),
            pa.field("section_path", pa.string()),
            pa.field("json_parent_id", pa.string()),
            pa.field("code_language", pa.string()),
            pa.field("char_offset", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])

        try:
            table = await asyncio.to_thread(
                db.create_table,
                collection,
                schema=schema,
                mode="overwrite",
            )
            # 创建全文索引
            await asyncio.to_thread(
                table.create_fts_index,
                "text",
                replace=True,
            )
            logger.info("collection_created", kb_id=kb_id, collection=collection, dim=dim)
        except Exception as e:
            raise StorageError(f"创建集合失败: {e}") from e

        return collection

    async def delete_collection(self, collection: str) -> None:
        db = await self._ensure_db()
        try:
            await asyncio.to_thread(db.drop_table, collection)
            logger.info("collection_deleted", collection=collection)
        except Exception as e:
            raise StorageError(f"删除集合失败: {e}") from e

    async def _get_table(self, collection: str) -> Table:
        db = await self._ensure_db()
        try:
            return await asyncio.to_thread(db.open_table, collection)
        except Exception as e:
            raise CollectionNotFoundError(collection) from e

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """写入/更新分块与向量。"""
        if not chunks:
            return

        table = await self._get_table(collection)

        records = []
        for chunk, vec in zip(chunks, vectors):
            records.append({
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "kb_id": chunk.kb_id,
                "seq": chunk.seq,
                "text": chunk.text,
                "content_type": chunk.content_type.value,
                "section_path": chunk.section_path,
                "json_parent_id": chunk.json_parent_id or "",
                "code_language": chunk.code_language or "",
                "char_offset": chunk.char_offset,
                "vector": vec,
            })

        try:
            # 先删除同 doc_id 的旧数据（增量更新场景）
            doc_ids = list({c.doc_id for c in chunks})
            for doc_id in doc_ids:
                try:
                    await asyncio.to_thread(
                        table.delete, f'doc_id = "{doc_id}"'
                    )
                except Exception:
                    pass  # 首次写入无旧数据

            await asyncio.to_thread(table.add, records)
            logger.info("chunks_upserted", collection=collection, count=len(records))
        except Exception as e:
            raise StorageError(f"写入失败: {e}") from e

    async def delete_by_doc(self, collection: str, doc_id: str) -> None:
        table = await self._get_table(collection)
        try:
            await asyncio.to_thread(table.delete, f'doc_id = "{doc_id}"')
            logger.info("doc_deleted", collection=collection, doc_id=doc_id)
        except Exception as e:
            raise StorageError(f"删除文档数据失败: {e}") from e

    async def vector_search(
        self,
        collection: str,
        query_vector: list[float],
        top_n: int = 50,
        filters: dict | None = None,
    ) -> list[dict]:
        """向量召回。"""
        table = await self._get_table(collection)

        try:
            query = table.search(query_vector).limit(top_n)
            if filters:
                for key, value in filters.items():
                    query = query.where(f'{key} = "{value}"')

            results = await asyncio.to_thread(query.to_list)
            return [
                {
                    "chunk_id": r["id"],
                    "doc_id": r["doc_id"],
                    "kb_id": r["kb_id"],
                    "text": r["text"],
                    "content_type": r["content_type"],
                    "section_path": r.get("section_path", ""),
                    "json_parent_id": r.get("json_parent_id", ""),
                    "code_language": r.get("code_language", ""),
                    "char_offset": r.get("char_offset", 0),
                    "score": float(r.get("_distance", 0)),
                    "search_type": "vector",
                }
                for r in results
            ]
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return []

    async def fts_search(
        self,
        collection: str,
        query: str,
        top_n: int = 50,
        filters: dict | None = None,
    ) -> list[dict]:
        """全文召回（BM25）。"""
        table = await self._get_table(collection)

        try:
            q = table.search(query, query_type="fts").limit(top_n)
            if filters:
                for key, value in filters.items():
                    q = q.where(f'{key} = "{value}"')

            results = await asyncio.to_thread(q.to_list)
            return [
                {
                    "chunk_id": r["id"],
                    "doc_id": r["doc_id"],
                    "kb_id": r["kb_id"],
                    "text": r["text"],
                    "content_type": r["content_type"],
                    "section_path": r.get("section_path", ""),
                    "json_parent_id": r.get("json_parent_id", ""),
                    "code_language": r.get("code_language", ""),
                    "char_offset": r.get("char_offset", 0),
                    "score": float(r.get("_score", 0)),
                    "search_type": "fts",
                }
                for r in results
            ]
        except Exception as e:
            logger.error("fts_search_failed", error=str(e))
            return []

    async def health_check(self) -> bool:
        try:
            db = await self._ensure_db()
            tables = await asyncio.to_thread(db.table_names)
            return True
        except Exception:
            return False
