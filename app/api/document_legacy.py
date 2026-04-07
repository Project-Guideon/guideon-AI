"""
[레거시] 직접 업로드 엔드포인트 - 테스트 용도로만 사용.
Core 연동 완료 후 이 파일 삭제 예정.

- POST /sites/{site_id}/documents/upload    : PDF 직접 업로드
- POST /sites/{site_id}/documents/upload_v2 : 구조화 RAG v2
- GET  /sites/{site_id}/documents/{doc_id}/status : 상태 조회
"""
import hashlib
import uuid
import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from psycopg.errors import UniqueViolation

from app.core.DB.PDF2db import create_doc_record, process_pdf_bytes
from app.core.DB.PDF2db_v2 import (
    create_doc_record as create_doc_record_v2,
    process_pdf_bytes_v2,
)
from app.core.DB.connect_db import get_conn

router = APIRouter()


def _check_duplicate(file_hash: str, site_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id FROM tb_document WHERE file_hash = %s AND site_id = %s",
                (file_hash, site_id),
            )
            return cur.fetchone()


def _get_doc_status(doc_id: int, site_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT doc_id, status, original_name, file_hash,
                          failed_reason, processed_at
                   FROM tb_document
                   WHERE doc_id = %s AND site_id = %s""",
                (doc_id, site_id),
            )
            return cur.fetchone()


@router.post("/sites/{site_id}/documents/upload")
async def upload_document(
    site_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    purge_old_chunks: bool = Form(True),
):
    pdf_bytes = await file.read()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    original_name = file.filename or "unknown"
    created_at = datetime.now(timezone.utc).isoformat()

    existing = await asyncio.to_thread(_check_duplicate, file_hash, site_id)
    if existing:
        return JSONResponse(
            {
                "success": False,
                "data": {"doc_id": existing[0]},
                "error": {
                    "code": "DOC_HASH_DUPLICATE",
                    "message": "동일한 파일이 이미 업로드되어 있습니다.",
                },
                "trace_id": str(uuid.uuid4()),
            },
            status_code=409,
        )

    try:
        doc_id = await asyncio.to_thread(
            create_doc_record,
            original_name=original_name,
            file_hash=file_hash,
            site_id=site_id,
            file_size=len(pdf_bytes),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except UniqueViolation:
        return JSONResponse(
            {
                "success": False,
                "data": None,
                "error": {
                    "code": "DOC_HASH_DUPLICATE",
                    "message": "동일한 파일이 이미 업로드되어 있습니다.",
                },
                "trace_id": str(uuid.uuid4()),
            },
            status_code=409,
        )

    background_tasks.add_task(
        process_pdf_bytes,
        doc_id=doc_id,
        pdf_bytes=pdf_bytes,
        site_id=site_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        purge_old_chunks=purge_old_chunks,
    )

    return JSONResponse({
        "success": True,
        "data": {
            "doc_id": doc_id,
            "status": "PENDING",
            "original_name": original_name,
            "file_hash": file_hash,
            "created_at": created_at,
        },
        "error": None,
        "trace_id": str(uuid.uuid4()),
    })


@router.get("/sites/{site_id}/documents/{doc_id}/status")
async def get_document_status(site_id: int, doc_id: int):
    row = await asyncio.to_thread(_get_doc_status, doc_id, site_id)

    if not row:
        return JSONResponse(
            {
                "success": False,
                "data": None,
                "error": "문서를 찾을 수 없습니다.",
                "trace_id": str(uuid.uuid4()),
            },
            status_code=404,
        )

    doc_id_, status, original_name, file_hash, failed_reason, processed_at = row
    return JSONResponse({
        "success": True,
        "data": {
            "doc_id": doc_id_,
            "status": status,
            "original_name": original_name,
            "file_hash": file_hash,
            "failed_reason": failed_reason,
            "processed_at": processed_at.isoformat() if processed_at else None,
        },
        "error": None,
        "trace_id": str(uuid.uuid4()),
    })


@router.post("/sites/{site_id}/documents/upload_v2")
async def upload_document_v2(
    site_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    purge_old_chunks: bool = Form(True),
):
    pdf_bytes = await file.read()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    original_name = file.filename or "unknown"
    created_at = datetime.now(timezone.utc).isoformat()

    existing = await asyncio.to_thread(_check_duplicate, file_hash, site_id)
    if existing:
        return JSONResponse(
            {
                "success": False,
                "data": {"doc_id": existing[0]},
                "error": {
                    "code": "DOC_HASH_DUPLICATE",
                    "message": "동일한 파일이 이미 업로드되어 있습니다.",
                },
                "trace_id": str(uuid.uuid4()),
            },
            status_code=409,
        )

    try:
        doc_id = await asyncio.to_thread(
            create_doc_record_v2,
            original_name=original_name,
            file_hash=file_hash,
            site_id=site_id,
            file_size=len(pdf_bytes),
        )
    except UniqueViolation:
        return JSONResponse(
            {
                "success": False,
                "data": None,
                "error": {
                    "code": "DOC_HASH_DUPLICATE",
                    "message": "동일한 파일이 이미 업로드되어 있습니다.",
                },
                "trace_id": str(uuid.uuid4()),
            },
            status_code=409,
        )

    background_tasks.add_task(
        process_pdf_bytes_v2,
        doc_id=doc_id,
        pdf_bytes=pdf_bytes,
        site_id=site_id,
        purge_old_chunks=purge_old_chunks,
    )

    return JSONResponse({
        "success": True,
        "data": {
            "doc_id": doc_id,
            "status": "PENDING",
            "original_name": original_name,
            "file_hash": file_hash,
            "created_at": created_at,
            "pipeline_version": "v2",
        },
        "error": None,
        "trace_id": str(uuid.uuid4()),
    })
