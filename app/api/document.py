"""
Core 연동 문서 처리 엔드포인트

- POST /v1/documents/process : Core에서 doc_id + storage_url 받아 처리 후 콜백
"""
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from app.core.services.document_processor import process_pdf_from_url

router = APIRouter()


class ProcessDocumentRequest(BaseModel):
    doc_id: int
    site_id: int
    storage_url: str


@router.post("/v1/documents/process")
async def process_document(req: ProcessDocumentRequest, background_tasks: BackgroundTasks):
    """
    Core에서 호출. 문서를 백그라운드에서 처리하고 완료 후 Core에 콜백.
    즉시 200 반환 (fire-and-forget).
    """
    background_tasks.add_task(
        process_pdf_from_url,
        doc_id=req.doc_id,
        site_id=req.site_id,
        storage_url=req.storage_url,
    )
    return {"ok": True}
