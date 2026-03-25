"""
Core 연동 문서 처리 엔드포인트

- POST /v1/documents/process : Core에서 doc_id + storage_url 받아 처리 후 콜백
"""
import os
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from app.core.services.document_processor import process_pdf_from_url

router = APIRouter()

# 내부 서비스 간 호출 인증 토큰 (.env에 설정)
_INTERNAL_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")

# storage_url 허용 호스트 (SSRF 방지)
_ALLOWED_STORAGE_HOSTS = {"admin-bff", "host.docker.internal", "localhost"}


class ProcessDocumentRequest(BaseModel):
    doc_id: int
    site_id: int
    storage_url: str


@router.post("/v1/documents/process")
async def process_document(
    req: ProcessDocumentRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: str | None = Header(default=None),
):
    """
    Core에서 호출. 문서를 백그라운드에서 처리하고 완료 후 Core에 콜백.
    즉시 200 반환 (fire-and-forget).
    """
    # 호출자 검증 (INTERNAL_API_TOKEN 미설정 시 개발 환경으로 간주해 통과)
    if _INTERNAL_TOKEN and x_internal_token != _INTERNAL_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")

    # storage_url 호스트 allowlist 검증 (SSRF 방지)
    parsed = urlparse(req.storage_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _ALLOWED_STORAGE_HOSTS:
        raise HTTPException(status_code=400, detail="invalid storage_url")

    background_tasks.add_task(
        process_pdf_from_url,
        doc_id=req.doc_id,
        site_id=req.site_id,
        storage_url=req.storage_url,
    )
    return {"ok": True}
