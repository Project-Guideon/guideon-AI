"""
WebSocket 스트리밍 엔드포인트
- WS /ws/stream : 오디오 스트림 수신 → STT → LLM → 텍스트 응답
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langsmith import trace as ls_trace

from app.core.dependencies import stt, traced_text_run

router = APIRouter()


def _new_trace_id() -> str:
    import uuid
    return str(uuid.uuid4())


async def audio_receiver(websocket: WebSocket, audio_q: "asyncio.Queue[Optional[bytes]]"):
    """
    클라가 보내는 프레임:
      - text: {"type":"stop"} 같은 control
      - bytes: PCM chunk
    """
    while True:
        msg = await websocket.receive()

        if msg.get("text") is not None:
            data = json.loads(msg["text"])
            if data.get("type") == "stop":
                await audio_q.put(None)  # 종료 신호
                return
            continue

        if msg.get("bytes") is not None:
            await audio_q.put(msg["bytes"])


@router.websocket("/ws/stream")
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
