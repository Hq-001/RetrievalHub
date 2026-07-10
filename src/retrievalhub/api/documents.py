"""REST 文档管理路由 - 上传、查询、删除、重试、知识库管理。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from retrievalhub.core.exceptions import DuplicateDocumentError, UnsupportedFormatError
from retrievalhub.core.models import DocFormat, DocStatus, Document, KnowledgeBase
from retrievalhub.ingest.file_handler import (
    compute_content_hash,
    detect_format,
    save_original_file,
    validate_content,
)

router = APIRouter(prefix="/v1", tags=["documents"])

# 全局依赖注入位（由 app.py 设置）
_storage = None
_metadata = None
_embedder = None
_pipeline = None
_enqueue = None
_original_dir: Path = Path("./data/originals")


def set_dependencies(
    storage, metadata, embedder, pipeline, enqueue, original_dir: Path
) -> None:
    """注入依赖（由 app.py 在启动时调用）。"""
    global _storage, _metadata, _embedder, _pipeline, _enqueue, _original_dir
    _storage = storage
    _metadata = metadata
    _embedder = embedder
    _pipeline = pipeline
    _enqueue = enqueue
    _original_dir = original_dir


# ---- 知识库管理 ----


@router.post("/knowledge-bases", status_code=status.HTTP_201_CREATED)
async def create_kb(
    name: str = Form(...),
    description: str = Form(""),
):
    kb_id = str(uuid.uuid4())
    kb = KnowledgeBase(
        id=kb_id,
        name=name,
        description=description,
        embedding_model=_embedder._model if hasattr(_embedder, "_model") else "mock",
        embedding_dim=_embedder.dim,
    )
    _metadata.create_kb(kb)
    collection = await _storage.create_collection(kb_id, dim=_embedder.dim)
    _metadata.update_kb_collection(kb_id, collection)
    return {"id": kb_id, "name": name, "collection": collection}


@router.get("/knowledge-bases")
async def list_kbs():
    kbs = _metadata.list_kbs()
    return {"knowledge_bases": [kb.model_dump() for kb in kbs]}


@router.delete("/knowledge-bases/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(kb_id: str):
    kb = _metadata.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if kb.active_collection:
        await _storage.delete_collection(kb.active_collection)
    _metadata.delete_kb(kb_id)
    return None


# ---- 文档管理 ----


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
):
    # 校验知识库
    kb = _metadata.get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # 读取文件内容
    content_bytes = await file.read()
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    # 格式校验
    try:
        fmt = detect_format(file.filename)
    except UnsupportedFormatError:
        raise HTTPException(status_code=400, detail="Only .md/.json/.jsonl supported")

    # 大小校验
    if len(content_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # 内容校验
    try:
        validate_content(content, fmt)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 内容哈希
    content_hash = compute_content_hash(content)

    # 保存原文
    file_path = save_original_file(content, f"{uuid.uuid4().hex}_{file.filename}", _original_dir)

    # 创建文档记录
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        kb_id=kb_id,
        filename=file.filename,
        file_format=fmt,
        file_size=len(content_bytes),
        content_hash=content_hash,
    )

    try:
        _metadata.create_document(doc)
    except DuplicateDocumentError as e:
        raise HTTPException(
            status_code=409,
            detail=f"Document already exists: {e.existing_doc_id}",
        )

    # 异步入库
    collection = kb.active_collection
    await _enqueue.submit(
        _pipeline.ingest,
        doc_id,
        file_path,
        kb_id,
        content_hash,
        collection,
        task_id=doc_id,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "doc_id": doc_id,
            "status": "processing",
            "status_url": f"/v1/documents/{doc_id}",
            "filename": file.filename,
            "format_version": doc.format_version,
        },
    )


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = _metadata.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc.id,
        "kb_id": doc.kb_id,
        "filename": doc.filename,
        "file_format": doc.file_format.value,
        "file_size": doc.file_size,
        "content_hash": doc.content_hash,
        "status": doc.status.value,
        "chunk_count": doc.chunk_count,
        "error_message": doc.error_message,
        "format_version": doc.format_version,
        "attempt_count": doc.attempt_count,
        "created_at": doc.created_at.isoformat(),
    }


@router.get("/knowledge-bases/{kb_id}/documents")
async def list_documents(kb_id: str, status_filter: str | None = None):
    docs = _metadata.list_documents(kb_id, status=status_filter)
    return {
        "documents": [
            {
                "doc_id": d.id,
                "filename": d.filename,
                "status": d.status.value,
                "chunk_count": d.chunk_count,
                "file_format": d.file_format.value,
            }
            for d in docs
        ]
    }


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str):
    doc = _metadata.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    kb = _metadata.get_kb(doc.kb_id)
    if kb and kb.active_collection:
        await _storage.delete_by_doc(kb.active_collection, doc_id)

    _metadata.delete_document(doc_id)
    return None


@router.post("/documents/{doc_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_document(doc_id: str):
    doc = _metadata.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != DocStatus.FAILED:
        raise HTTPException(status_code=409, detail="Only failed documents can be retried")

    # 更新状态为 processing（attempt_count +1）
    _metadata.update_document_status(doc_id, DocStatus.PROCESSING)

    # 重新查找原文路径（通过 content_hash 重建）
    # 实际场景中需要记录 file_path，这里简化为重新读取
    kb = _metadata.get_kb(doc.kb_id)
    if not kb or not kb.active_collection:
        raise HTTPException(status_code=500, detail="Knowledge base collection not found")

    # 查找原文（通过文件名前缀匹配）
    original_files = list(_original_dir.glob(f"*_{doc.filename}"))
    if not original_files:
        raise HTTPException(status_code=404, detail="Original file not found")

    file_path = original_files[0]

    # 重新提交入库
    await _enqueue.submit(
        _pipeline.ingest,
        doc_id,
        file_path,
        doc.kb_id,
        doc.content_hash,
        kb.active_collection,
        task_id=doc_id,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "doc_id": doc_id,
            "status": "processing",
            "status_url": f"/v1/documents/{doc_id}",
        },
    )
