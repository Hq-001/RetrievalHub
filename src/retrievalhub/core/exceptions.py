"""统一异常体系。"""

from __future__ import annotations


class RetrievalHubError(Exception):
    """所有自定义异常的基类。"""

    def __init__(self, message: str, code: str = "UNKNOWN") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# ---- 解析类 ----


class ParseError(RetrievalHubError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="PARSE_ERROR")


class UnsupportedFormatError(RetrievalHubError):
    def __init__(self, fmt: str) -> None:
        super().__init__(f"不支持的格式: {fmt}", code="UNSUPPORTED_FORMAT")


# ---- 存储类 ----


class StorageError(RetrievalHubError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="STORAGE_ERROR")


class CollectionNotFoundError(StorageError):
    def __init__(self, collection: str) -> None:
        super().__init__(f"集合不存在: {collection}")


# ---- 入库类 ----


class IngestError(RetrievalHubError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="INGEST_ERROR")


class DuplicateDocumentError(RetrievalHubError):
    """并发上传相同内容时触发 (409)。"""

    def __init__(self, doc_id: str, message: str = "文档已存在") -> None:
        super().__init__(message, code="DUPLICATE_DOCUMENT")
        self.existing_doc_id = doc_id


class IngestTimeoutError(IngestError):
    def __init__(self, doc_id: str, timeout_sec: int) -> None:
        super().__init__(
            f"入库超时: doc={doc_id}, timeout={timeout_sec}s"
        )


# ---- 嵌入类 ----


class EmbeddingError(RetrievalHubError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="EMBEDDING_ERROR")


# ---- 检索类 ----


class SearchError(RetrievalHubError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="SEARCH_ERROR")


class SearchTimeoutError(SearchError):
    def __init__(self, timeout_ms: int) -> None:
        super().__init__(f"检索超时: {timeout_ms}ms")
