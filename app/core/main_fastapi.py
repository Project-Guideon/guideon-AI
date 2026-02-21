# main_fastapi.py
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import Response, JSONResponse
from dotenv import load_dotenv

from app.core.services.stt_google import GoogleSTT, STTConfig
from app.core.services.tts_google import GoogleTTS, TTSConfig
from app.core.services.rag_pgvector import PgVectorRAG
from app.core.services.llm_openai import OpenAILLM, LLMConfig
from app.core.services.pipeline import VoicePipeline
from app.core.DB.PDF2db import create_doc_record, process_pdf_bytes
from app.core.DB.connect_db import get_conn


load_dotenv()

app = FastAPI(title="Guideon Voice QA")

stt = GoogleSTT(STTConfig(language_code="ko-KR", sample_rate_hz=16000))
tts = GoogleTTS(TTSConfig(language_code="ko-KR"))
rag = PgVectorRAG(model_name="paraphrase-multilingual-mpnet-base-v2")
llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", temperature=0.7, max_tokens=500))

pipeline = VoicePipeline(stt=stt, rag=rag, llm=llm, tts=tts)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"ok": True, "msg": "Go to /docs"}


@app.post("/voice_qa")
async def voice_qa(audio: UploadFile = File(...), site_id: int = 1, k: int = 5):
    audio_bytes = await audio.read()
    result = pipeline.run(audio_bytes, site_id=site_id, k=k)

    return Response(
        content=result.voice_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=answer.mp3"},
    )


@app.post("/text_qa") # 텍스트로 들어왔을때 택스트 답변 과정 사용할지 안할지는 미정
async def text_qa(query: str, site_id: int = 1, k: int = 5):
    contexts = rag.retrieve(query=query, site_id=site_id, k=k)
    answer = llm.generate(query=query, contexts=contexts)
    return JSONResponse({"query": query, "answer": answer})


@app.post("/sites/{site_id}/documents/upload")
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
    original_name = file.filename
    created_at = datetime.now(timezone.utc).isoformat()

    # 409: 동일 site에 같은 파일(해시) 중복 업로드 체크
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id FROM tb_document WHERE file_hash = %s AND site_id = %s",
                (file_hash, site_id),
            )
            existing = cur.fetchone()

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
            #파일명이 달라도 내용이 같으면 같은 해시값=>409error발생
            status_code=409,
        )

    doc_id = create_doc_record(
        original_name=original_name,
        file_hash=file_hash,
        site_id=site_id,
        file_size=len(pdf_bytes),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        purge_old_chunks=purge_old_chunks,
    )

    # 즉시 응답 후 백그라운드에서 청킹/임베딩/저장 처리
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

# 문서 처리 상태 조회 API
@app.get("/sites/{site_id}/documents/{doc_id}/status")
async def get_document_status(site_id: int, doc_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT doc_id, status, original_name, file_hash,
                          failed_reason, processed_at
                   FROM tb_document
                   WHERE doc_id = %s AND site_id = %s""",
                (doc_id, site_id),
            )
            row = cur.fetchone()

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
