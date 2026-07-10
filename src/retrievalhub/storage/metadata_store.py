"""SQLite 元数据库 - 文档/分块元信息持久化。

包含：
- documents 表：文档元信息 + 状态机
- knowledge_bases 表：知识库元信息
- UNIQUE 约束：(knowledge_base_id, content_hash) 防并发去重
- 崩溃恢复字段：started_at, attempt_count
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrievalhub.core.exceptions import DuplicateDocumentError
from retrievalhub.core.models import DocFormat, DocStatus, Document, KnowledgeBase
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetadataStore:
    """SQLite 元数据存储。"""

    def __init__(self, db_path: str = "./data/metadata.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self._ensure_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                embedding_model TEXT DEFAULT '',
                embedding_dim INTEGER DEFAULT 0,
                active_collection TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_format TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                content_hash TEXT NOT NULL,
                status TEXT DEFAULT 'processing',
                started_at TEXT,
                attempt_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                error_message TEXT,
                format_version TEXT DEFAULT '1',
                created_at TEXT NOT NULL,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id),
                UNIQUE (kb_id, content_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_kb_id ON documents(kb_id);
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
        """)
        conn.commit()

    # ---- 知识库 ----

    def create_kb(self, kb: KnowledgeBase) -> None:
        conn = self._ensure_conn()
        conn.execute(
            """INSERT INTO knowledge_bases (id, name, description, embedding_model,
               embedding_dim, active_collection, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                kb.id, kb.name, kb.description, kb.embedding_model,
                kb.embedding_dim, kb.active_collection,
                kb.created_at.isoformat(),
            ),
        )
        conn.commit()

    def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        conn = self._ensure_conn()
        row = conn.execute(
            "SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)
        ).fetchone()
        if not row:
            return None
        return KnowledgeBase(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            embedding_model=row["embedding_model"],
            embedding_dim=row["embedding_dim"],
            active_collection=row["active_collection"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_kbs(self) -> list[KnowledgeBase]:
        conn = self._ensure_conn()
        rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC").fetchall()
        return [
            KnowledgeBase(
                id=r["id"], name=r["name"], description=r["description"],
                embedding_model=r["embedding_model"], embedding_dim=r["embedding_dim"],
                active_collection=r["active_collection"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def delete_kb(self, kb_id: str) -> None:
        conn = self._ensure_conn()
        conn.execute("DELETE FROM documents WHERE kb_id = ?", (kb_id,))
        conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        conn.commit()

    def update_kb_collection(self, kb_id: str, collection: str) -> None:
        conn = self._ensure_conn()
        conn.execute(
            "UPDATE knowledge_bases SET active_collection = ? WHERE id = ?",
            (collection, kb_id),
        )
        conn.commit()

    # ---- 文档 ----

    def create_document(self, doc: Document) -> None:
        """创建文档记录。UNIQUE 约束冲突时抛 DuplicateDocumentError。"""
        conn = self._ensure_conn()
        try:
            conn.execute(
                """INSERT INTO documents (id, kb_id, filename, file_format, file_size,
                   content_hash, status, started_at, attempt_count, chunk_count,
                   error_message, format_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.id, doc.kb_id, doc.filename, doc.file_format.value,
                    doc.file_size, doc.content_hash, doc.status.value,
                    doc.started_at.isoformat() if doc.started_at else None,
                    doc.attempt_count, doc.chunk_count,
                    doc.error_message, doc.format_version,
                    doc.created_at.isoformat(),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            if "UNIQUE" in str(e):
                # 查找已存在的文档
                row = conn.execute(
                    "SELECT id FROM documents WHERE kb_id = ? AND content_hash = ?",
                    (doc.kb_id, doc.content_hash),
                ).fetchone()
                existing_id = row["id"] if row else ""
                raise DuplicateDocumentError(existing_id) from e
            raise

    def get_document(self, doc_id: str) -> Document | None:
        conn = self._ensure_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def list_documents(self, kb_id: str, status: str | None = None) -> list[Document]:
        conn = self._ensure_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND status = ? ORDER BY created_at DESC",
                (kb_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC",
                (kb_id,),
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def update_document_status(
        self,
        doc_id: str,
        status: DocStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        conn = self._ensure_conn()
        sets = ["status = ?"]
        params: list[Any] = [status.value]

        if chunk_count is not None:
            sets.append("chunk_count = ?")
            params.append(chunk_count)
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)
        if status == DocStatus.PROCESSING:
            sets.append("started_at = ?")
            params.append(_utcnow_iso())
            sets.append("attempt_count = attempt_count + 1")

        params.append(doc_id)
        conn.execute(
            f"UPDATE documents SET {', '.join(sets)} WHERE id = ?", params
        )
        conn.commit()

    def delete_document(self, doc_id: str) -> None:
        conn = self._ensure_conn()
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

    # ---- 崩溃恢复 ----

    def get_stale_processing_docs(self, timeout_sec: int) -> list[Document]:
        """获取超时仍 processing 的文档（崩溃恢复用）。"""
        conn = self._ensure_conn()
        threshold = timeout_sec * 2
        rows = conn.execute(
            """SELECT * FROM documents
               WHERE status = 'processing'
               AND started_at IS NOT NULL
               AND julianday('now') - julianday(started_at) > ? / 86400.0""",
            (threshold,),
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    # ---- 辅助 ----

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            kb_id=row["kb_id"],
            filename=row["filename"],
            file_format=DocFormat(row["file_format"]),
            file_size=row["file_size"],
            content_hash=row["content_hash"],
            status=DocStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            attempt_count=row["attempt_count"],
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            format_version=row["format_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
