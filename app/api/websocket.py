"""
WebSocket 스트리밍 엔드포인트
- WS /ws/stream : 오디오 스트림 수신 → STT → text_qa와 동일한 LangGraph 파이프라인 → TTS 스트리밍 응답
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langsmith import trace as ls_trace, traceable

from app.core.dependencies import stt, tts, traced_text_run

router = APIRouter()


def _new_trace_id() -> str:
    return str(uuid.uuid4())


def map_tts_language(lang2: str) -> str:
    lang2 = (lang2 or "ko").split("-")[0].lower()
    mapping = {
        "ko": "ko-KR",
        "en": "en-US",
        "ja": "ja-JP",
        "zh": "cmn-CN",
    }
    return mapping.get(lang2, "ko-KR")


def ms_delta(start: float | None, end: float | None) -> int | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000)


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s*|(?<=요\.)\s*|(?<=니다\.)\s*", text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_mascot_payload(raw_mascot: object) -> dict:
    """
    text_qa에서 _build_mascot_dict()가 만드는 형태와 최대한 맞춤.
    start 메시지에서 mascot이 없더라도 빈 문자열 키들로 맞춰줌.
    """
    if not isinstance(raw_mascot, dict):
        raw_mascot = {}

    return {
        "system_prompt": raw_mascot.get("system_prompt", "") or "",
        "mascot_name": raw_mascot.get("mascot_name", "") or "",
        "mascot_greeting": raw_mascot.get("mascot_greeting", "") or "",
        "mascot_base_persona": raw_mascot.get("mascot_base_persona", "") or "",
        "mascot_smalltalk_style": raw_mascot.get("mascot_smalltalk_style", "") or "",
        "mascot_struct_db_style": raw_mascot.get("mascot_struct_db_style", "") or "",
        "mascot_RAG_style": raw_mascot.get("mascot_RAG_style", "") or "",
        "mascot_event_style": raw_mascot.get("mascot_event_style", "") or "",
    }


@traceable(name="emit_latency_summary", run_type="tool")
def emit_latency_summary(
    trace_id: str,
    site_id: int,
    realtime: bool,
    tts_stream: bool,
    stt_latency_ms: int | None,
    qa_first_sentence_ms: int | None,
    qa_total_ms: int | None,
    tts_first_audio_latency_ms: int | None,
    tts_total_ms: int | None,
    tts_sentence_count: int,
    language_code: str,
):
    return {
        "trace_id": trace_id,
        "site_id": site_id,
        "realtime": realtime,
        "tts_stream": tts_stream,
        "language_code": language_code,
        "stt_latency_ms": stt_latency_ms,
        "qa_first_sentence_latency_ms": qa_first_sentence_ms,
        "qa_total_ms": qa_total_ms,
        "tts_first_audio_latency_ms": tts_first_audio_latency_ms,
        "tts_total_ms": tts_total_ms,
        "tts_sentence_count": tts_sentence_count,
    }


@traceable(name="tts_synthesize", run_type="tool")
def traced_tts_synthesize(sentence: str, tts_language_code: str):
    return tts.synthesize(sentence, tts_language_code)


@traceable(name="ws_text_qa_invoke", run_type="chain")
def traced_ws_text_qa_invoke(
    query: str,
    site_id: int,
    language_code: str,
    mascot: dict | None = None,
):
    """
    반드시 text_qa와 동일한 traced_text_run 경로를 사용.
    """
    mascot = normalize_mascot_payload(mascot)
    return traced_text_run(query, site_id, language_code, mascot)


async def audio_receiver(
    websocket: WebSocket,
    audio_q: "asyncio.Queue[Optional[bytes]]",
    timing: dict,
):
    """
    클라가 보내는 프레임:
    - text: {"type":"stop"} 같은 control
    - bytes: PCM chunk
    """
    try:
        while True:
            msg = await websocket.receive()

            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                msg_type = data.get("type")

                if msg_type == "stop":
                    timing["stop_received_at"] = time.perf_counter()
                    await audio_q.put(None)
                    return

                continue

            if msg.get("bytes") is not None:
                if timing.get("first_audio_at") is None:
                    timing["first_audio_at"] = time.perf_counter()
                timing["last_audio_at"] = time.perf_counter()
                await audio_q.put(msg["bytes"])

    except WebSocketDisconnect:
        await audio_q.put(None)
        return


async def send_tts_chunks(
    websocket: WebSocket,
    sentences: list[str],
    tts_language_code: str,
    trace_id: str,
    start_seq: int = 0,
) -> dict:
    non_empty_sentences = [s.strip() for s in sentences if s and s.strip()]

    first_tts_audio_at = None
    total_tts_ms = 0
    current_seq = start_seq

    for idx, sentence in enumerate(non_empty_sentences):
        t0 = time.perf_counter()

        audio_chunk = await asyncio.to_thread(
            traced_tts_synthesize,
            sentence,
            tts_language_code,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        total_tts_ms += elapsed_ms

        if first_tts_audio_at is None:
            first_tts_audio_at = time.perf_counter()

        await websocket.send_text(json.dumps({
            "type": "tts_chunk",
            "seq": current_seq,
            "text": sentence,
            "audio_format": "mp3",
            "audio_b64": base64.b64encode(audio_chunk).decode("utf-8"),
            "language_code": tts_language_code,
            "is_final": idx == len(non_empty_sentences) - 1,
            "trace_id": trace_id,
        }, ensure_ascii=False))

        current_seq += 1

    return {
        "first_tts_audio_at": first_tts_audio_at,
        "total_tts_ms": total_tts_ms,
        "sentence_count": len(non_empty_sentences),
        "next_seq": current_seq,
    }


def extract_answer_text(result) -> str:
    """
    traced_text_run 반환 형태 방어적으로 처리
    """
    if result is None:
        return ""

    if isinstance(result, dict):
        return (result.get("answer") or result.get("answer_text") or "").strip()

    answer = getattr(result, "answer", None)
    if isinstance(answer, str):
        return answer.strip()

    answer_text = getattr(result, "answer_text", None)
    if isinstance(answer_text, str):
        return answer_text.strip()

    return ""


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
            }, ensure_ascii=False))
            await websocket.close()
            return

        site_id = int(start.get("site_id", 1))
        stt_language = start.get("language_code", "ko-KR")
        sample_rate_hz = int(start.get("sample_rate_hz", 16000))
        interim_results = bool(start.get("interim_results", True))
        tts_stream = bool(start.get("tts_stream", True))
        realtime = bool(start.get("realtime", True))
        mascot = normalize_mascot_payload(start.get("mascot"))

        with ls_trace(
            name="ws_voice_textqa_pipeline",
            run_type="chain",
            metadata={
                "trace_id": trace_id,
                "site_id": site_id,
                "tts_stream": tts_stream,
                "realtime": realtime,
            },
            inputs={
                "site_id": site_id,
                "stt_language": stt_language,
                "sample_rate_hz": sample_rate_hz,
                "interim_results": interim_results,
                "tts_stream": tts_stream,
                "realtime": realtime,
                "mascot": mascot,
            },
        ):
            audio_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
            timing = {
                "ws_start_at": time.perf_counter(),
                "first_audio_at": None,
                "last_audio_at": None,
                "stop_received_at": None,
                "stt_final_at": None,
                "qa_start_at": None,
                "qa_first_sentence_at": None,
                "tts_first_audio_at": None,
            }

            recv_task = asyncio.create_task(audio_receiver(websocket, audio_q, timing))

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "stt_start",
                "trace_id": trace_id,
            }, ensure_ascii=False))

            last_interim = ""
            last_final = ""
            last_lang_code = stt_language

            with ls_trace(
                name="stt_stream",
                run_type="tool",
                metadata={
                    "trace_id": trace_id,
                    "primary_language": stt_language,
                    "sample_rate_hz": sample_rate_hz,
                    "interim_results": interim_results,
                },
            ):
                async for ev in stt.stream_events(
                    audio_q,
                    primary_language=stt_language,
                    sample_rate_hz=sample_rate_hz,
                    interim_results=interim_results,
                    single_utterance=False,
                ):
                    last_lang_code = ev.language_code or last_lang_code

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

            timing["stt_final_at"] = time.perf_counter()

            if recv_task:
                await recv_task

            stt_latency_ms = ms_delta(
                timing["stop_received_at"],
                timing["stt_final_at"],
            )

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "stt_done",
                "trace_id": trace_id,
            }, ensure_ascii=False))

            query = (last_final or last_interim).strip()
            if not query:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "code": "EMPTY_TRANSCRIPT",
                    "message": "no transcript",
                    "trace_id": trace_id,
                }, ensure_ascii=False))
                await websocket.close()
                return

            tts_language_code = map_tts_language(last_lang_code)

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "graph_start",
                "trace_id": trace_id,
            }, ensure_ascii=False))

            timing["qa_start_at"] = time.perf_counter()

            qa_result = await asyncio.to_thread(
                traced_ws_text_qa_invoke,
                query,
                site_id,
                last_lang_code,
                mascot,
            )

            qa_total_ms = ms_delta(
                timing["qa_start_at"],
                time.perf_counter(),
            )

            answer_text = extract_answer_text(qa_result)
            if not answer_text:
                answer_text = "죄송해요. 답변을 생성하지 못했어요."

            sentences = split_sentences(answer_text)

            tts_first_audio_latency_ms = None
            tts_total_ms = None
            tts_sentence_count = 0

            if realtime:
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "stage": "graph_stream_start",
                    "trace_id": trace_id,
                }, ensure_ascii=False))

                for idx, sentence in enumerate(sentences):
                    if timing["qa_first_sentence_at"] is None:
                        timing["qa_first_sentence_at"] = time.perf_counter()

                    await websocket.send_text(json.dumps({
                        "type": "llm_sentence",
                        "text": sentence,
                        "language_code": last_lang_code,
                        "trace_id": trace_id,
                    }, ensure_ascii=False))

                    if tts_stream:
                        if idx == 0:
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "stage": "tts_start",
                                "trace_id": trace_id,
                            }, ensure_ascii=False))

                        tts_result = await send_tts_chunks(
                            websocket=websocket,
                            sentences=[sentence],
                            tts_language_code=tts_language_code,
                            trace_id=trace_id,
                            start_seq=idx,
                        )

                        tts_total_ms = (tts_total_ms or 0) + tts_result["total_tts_ms"]
                        tts_sentence_count += tts_result["sentence_count"]

                        if timing["tts_first_audio_at"] is None:
                            timing["tts_first_audio_at"] = tts_result["first_tts_audio_at"]

                qa_first_sentence_ms = ms_delta(
                    timing["qa_start_at"],
                    timing["qa_first_sentence_at"],
                )

                if tts_stream and timing["tts_first_audio_at"] is not None:
                    tts_first_audio_latency_ms = ms_delta(
                        timing["qa_start_at"],
                        timing["tts_first_audio_at"],
                    )

                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "stage": "tts_done",
                        "trace_id": trace_id,
                    }, ensure_ascii=False))

                await websocket.send_text(json.dumps({
                    "type": "final_text",
                    "site_id": site_id,
                    "language_code": last_lang_code,
                    "query": query,
                    "answer": answer_text,
                    "trace_id": trace_id,
                }, ensure_ascii=False))

                await websocket.send_text(json.dumps({
                    "type": "status",
                    "stage": "graph_stream_done",
                    "trace_id": trace_id,
                }, ensure_ascii=False))

                emit_latency_summary(
                    trace_id=trace_id,
                    site_id=site_id,
                    realtime=True,
                    tts_stream=tts_stream,
                    stt_latency_ms=stt_latency_ms,
                    qa_first_sentence_ms=qa_first_sentence_ms,
                    qa_total_ms=qa_total_ms,
                    tts_first_audio_latency_ms=tts_first_audio_latency_ms,
                    tts_total_ms=tts_total_ms if tts_stream else None,
                    tts_sentence_count=tts_sentence_count if tts_stream else 0,
                    language_code=tts_language_code,
                )

            else:
                await websocket.send_text(json.dumps({
                    "type": "final_text",
                    "site_id": site_id,
                    "language_code": last_lang_code,
                    "query": query,
                    "answer": answer_text,
                    "trace_id": trace_id,
                }, ensure_ascii=False))

                qa_first_sentence_ms = None

                if tts_stream and answer_text.strip():
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "stage": "tts_start",
                        "trace_id": trace_id,
                    }, ensure_ascii=False))

                    tts_result = await send_tts_chunks(
                        websocket=websocket,
                        sentences=sentences or [answer_text],
                        tts_language_code=tts_language_code,
                        trace_id=trace_id,
                        start_seq=0,
                    )

                    tts_total_ms = tts_result["total_tts_ms"]
                    tts_sentence_count = tts_result["sentence_count"]
                    tts_first_audio_latency_ms = ms_delta(
                        timing["qa_start_at"],
                        tts_result["first_tts_audio_at"],
                    )

                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "stage": "tts_done",
                        "trace_id": trace_id,
                    }, ensure_ascii=False))

                emit_latency_summary(
                    trace_id=trace_id,
                    site_id=site_id,
                    realtime=False,
                    tts_stream=tts_stream,
                    stt_latency_ms=stt_latency_ms,
                    qa_first_sentence_ms=qa_first_sentence_ms,
                    qa_total_ms=qa_total_ms,
                    tts_first_audio_latency_ms=tts_first_audio_latency_ms,
                    tts_total_ms=tts_total_ms,
                    tts_sentence_count=tts_sentence_count,
                    language_code=tts_language_code,
                )

            await websocket.send_text(json.dumps({
                "type": "done",
                "trace_id": trace_id,
            }, ensure_ascii=False))
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
            }, ensure_ascii=False))
        finally:
            await websocket.close()

    finally:
        if recv_task and not recv_task.done():
            recv_task.cancel()