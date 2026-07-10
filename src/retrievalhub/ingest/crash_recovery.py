"""崩溃恢复 - 进程重启后扫描超时 processing 文档。

启动时扫描 SQLite 中仍为 processing 且超过 INGEST_TIMEOUT_SEC * 2 的文档，
自动置为 failed（reason: process_restarted），避免永久卡死。
"""

from __future__ import annotations

from retrievalhub.core.models import DocStatus
from retrievalhub.storage.metadata_store import MetadataStore
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class CrashRecovery:
    """崩溃恢复处理器。"""

    def __init__(self, metadata: MetadataStore, timeout_sec: int = 30) -> None:
        self._metadata = metadata
        self._timeout_sec = timeout_sec

    def recover_stale_documents(self) -> int:
        """扫描并恢复超时的 processing 文档。

        Returns:
            被标记为 failed 的文档数量
        """
        stale_docs = self._metadata.get_stale_processing_docs(self._timeout_sec)

        if not stale_docs:
            logger.info("crash_recovery_no_stale_docs")
            return 0

        count = 0
        for doc in stale_docs:
            self._metadata.update_document_status(
                doc_id=doc.id,
                status=DocStatus.FAILED,
                error_message="process_restarted",
            )
            count += 1
            logger.info(
                "crash_recovery_marked_failed",
                doc_id=doc.id,
                kb_id=doc.kb_id,
                reason="process_restarted",
            )

        logger.info("crash_recovery_complete", recovered_count=count)
        return count
