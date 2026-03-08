# main_fastapi.py
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
import os
import asyncio
import json
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from psycopg.errors import UniqueViolation
from dotenv import load_dotenv
from langsmith import traceable, trace as ls_trace
from openai import OpenAI

from app.core.services.stt_google import GoogleSTT, STTConfig
from app.core.services.tts_google import GoogleTTS, TTSConfig
from app.core.services.rag_pgvector import PgVectorRAG, OpenAIEmbedder
from app.core.services.rag_pgvector_v2 import PgVectorRAG_V2
from app.core.services.llm_openai import OpenAILLM, LLMConfig
from app.core.services.pipeline import VoicePipeline, TextPipeline
from app.core.DB.PDF2db import create_doc_record, process_pdf_bytes
from app.core.DB.PDF2db_v2 import (
    create_doc_record as create_doc_record_v2,
    process_pdf_bytes_v2,
)
from app.core.DB.connect_db import get_conn


load_dotenv()

app = FastAPI(title="Guideon Voice QA")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

stt = GoogleSTT(STTConfig(primary_language="ko-KR", sample_rate_hz=16000))
tts = GoogleTTS(TTSConfig(language_code="ko-KR"))
embedder = OpenAIEmbedder(client=client, model="text-embedding-3-small")

# RAG 버전 전환: .env에 RAG_VERSION=v2 설정하면 구조화 RAG + Hybrid Search 사용
RAG_VERSION = os.getenv("RAG_VERSION", "v1")

if RAG_VERSION == "v2":
    rag = PgVectorRAG_V2(embedder=embedder)
else:
    rag = PgVectorRAG(embedder=embedder)

llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", temperature=0.7, max_tokens=500))

pipeline = VoicePipeline(stt=stt, rag=rag, llm=llm, tts=tts)
text_pipeline = TextPipeline(rag=rag, llm=llm)


def _new_trace_id() -> str:
    return str(uuid.uuid4())


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"ok": True, "msg": "Go to /docs"}


@traceable(name="voice_qa_pipeline")
def traced_voice_run(audio_bytes: bytes, site_id: int):
    return pipeline.run(audio_bytes, site_id=site_id)


@app.post("/voice_qa")
async def voice_qa(audio: UploadFile = File(...), site_id: int = 1):
    audio_bytes = await audio.read()
    result = await asyncio.to_thread(traced_voice_run, audio_bytes, site_id)
    return Response(
        content=result.voice_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=answer.mp3"},
    )


@traceable(name="text_qa_pipeline")
def traced_text_run(query: str, site_id: int, language_code: str):
    return text_pipeline.run(query=query, site_id=site_id, language_code=language_code)


@app.post("/text_qa")
async def text_qa(query: str, site_id: int = 1, language_code: str = "ko-KR"):
    result = await asyncio.to_thread(traced_text_run, query, site_id, language_code)
    return JSONResponse({"query": result.query, "answer": result.answer})


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
    original_name = file.filename or "unknown"
    created_at = datetime.now(timezone.utc).isoformat()

    # 409: 동일 site에 같은 파일(해시) 중복 업로드 체크
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


# ── v2 구조화 RAG 업로드 엔드포인트 ─────────────────────────────────────────

@app.post("/sites/{site_id}/documents/upload_v2")
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


# ── WebSocket streaming (feature/experiment) ─────────────────────────────────

async def audio_receiver(websocket: WebSocket, audio_q: "asyncio.Queue[Optional[bytes]]"):
    """
    클라가 보내는 프레임:
      - text: {"type":"stop"} 같은 control
      - bytes: PCM chunk
    """
    while True:
        msg = await websocket.receive()

        # 텍스트 프레임
        if msg.get("text") is not None:
            data = json.loads(msg["text"])
            if data.get("type") == "stop":
                await audio_q.put(None)  # 종료 신호
                return
            continue

        # 바이너리 프레임 (오디오 청크)
        if msg.get("bytes") is not None:
            await audio_q.put(msg["bytes"])


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    trace_id = _new_trace_id()

    recv_task: Optional[asyncio.Task] = None
    try:
        start = json.loads(await websocket.receive_text())
        if start.get("type") != "start":
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "BAD_REQUEST",
                "message": "first message must be {type:'start'}",
                "trace_id": trace_id,
            }))
            await websocket.close()
            return

        site_id = int(start.get("site_id", 1))
        stt_language = start.get("language_code", "ko-KR")
        sample_rate_hz = int(start.get("sample_rate_hz", 16000))
        interim_results = bool(start.get("interim_results", True))

        with ls_trace(name="ws_voice_pipeline", run_type="chain", metadata={"trace_id": trace_id, "site_id": site_id}):
            audio_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
            t_audio_end: list = []

            async def _receiver_with_timer():
                await audio_receiver(websocket, audio_q)
                t_audio_end.append(time.perf_counter())

            recv_task = asyncio.create_task(_receiver_with_timer())

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "stt_start",
                "trace_id": trace_id,
            }))

            last_interim = ""
            last_final = ""
            last_lang2 = "ko"

            async for ev in stt.stream_events(
                audio_q,
                primary_language=stt_language,
                sample_rate_hz=sample_rate_hz,
                interim_results=interim_results,
                single_utterance=False,
            ):
                last_lang2 = ev.language_code or last_lang2
                if ev.is_final:
                    last_final = ev.transcript
                else:
                    last_interim = ev.transcript

                await websocket.send_text(json.dumps({
                    "type": "stt_final" if ev.is_final else "stt_interim",
                    "text": ev.transcript,
                    "language_code": ev.language_code,
                    "confidence": ev.confidence,
                    "is_final": ev.is_final,
                    "trace_id": trace_id,
                }, ensure_ascii=False))

            t_stt_final = time.perf_counter()
            stt_latency_ms = round((t_stt_final - t_audio_end[0]) * 1000) if t_audio_end else None
            with ls_trace(name="stt_processing", run_type="tool",
                          metadata={"latency_ms": stt_latency_ms, "language": stt_language}):
                pass

            if recv_task:
                await recv_task

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "stt_done",
                "trace_id": trace_id,
            }))

            query = (last_final or last_interim).strip()
            if not query:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "code": "EMPTY_TRANSCRIPT",
                    "message": "no transcript",
                    "trace_id": trace_id,
                }))
                await websocket.close()
                return

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "llm_start",
                "trace_id": trace_id,
            }))

            result = await asyncio.to_thread(traced_text_run, query, site_id, last_lang2)

            await websocket.send_text(json.dumps({
                "type": "final_text",
                "site_id": site_id,
                "language_code": last_lang2,
                "query": result.query,
                "answer": result.answer,
                "trace_id": trace_id,
            }, ensure_ascii=False))

            await websocket.send_text(json.dumps({"type": "done", "trace_id": trace_id}))
            await websocket.close()

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "INTERNAL",
                "message": str(e),
                "trace_id": trace_id,
            }))
        finally:
            await websocket.close()
    finally:
        if recv_task and not recv_task.done():
            recv_task.cancel()