"""
QA 엔드포인트
- POST /voice_qa  : 음성 파일 → 답변 음성 반환
- POST /text_qa   : 텍스트 질문 → 텍스트 답변 반환
"""
import asyncio

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from app.core.dependencies import traced_voice_run, traced_text_run

router = APIRouter()


@router.post("/voice_qa")
async def voice_qa(audio: UploadFile = File(...), site_id: int = 1):
    audio_bytes = await audio.read()
    result = await asyncio.to_thread(traced_voice_run, audio_bytes, site_id)
    return Response(
        content=result.voice_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=answer.mp3"},
    )


class TextQARequest(BaseModel):
    query: str
    site_id: int = 1
    language_code: str = "ko-KR"


@router.post("/text_qa")
async def text_qa(req: TextQARequest):
    result = await asyncio.to_thread(traced_text_run, req.query, req.site_id, req.language_code)
    return JSONResponse({"query": result.query, "answer": result.answer})
