"""
WebSocket 스트리밍 엔드포인트
- WS /ws/stream : 오디오 스트림 수신 → STT → LLM → TTS 스트리밍 응답
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Optional, AsyncIterator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langsmith import trace as ls_trace, traceable

from app.core.dependencies import stt, tts, rag, llm, traced_text_run

router = APIRouter()


def _new_trace_id() -> str:
    return str(uuid.uuid4())


def map_tts_language(lang2: str) -> str:
    """
    STT 결과(ko/en/ja/zh 등)를 Google TTS용 BCP-47 코드로 변환
    """
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


@traceable(name="emit_latency_summary", run_type="tool")
def emit_latency_summary(
    trace_id: str,
    site_id: int,
    realtime: bool,
    tts_stream: bool,
    stt_latency_ms: int | None,
    llm_first_sentence_ms: int | None,
    llm_total_ms: int | None,
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
        "llm_first_sentence_latency_ms": llm_first_sentence_ms,
        "llm_total_ms": llm_total_ms,
        "tts_first_audio_latency_ms": tts_first_audio_latency_ms,
        "tts_total_ms": tts_total_ms,
        "tts_sentence_count": tts_sentence_count,
    }


@traceable(name="rag_retrieve", run_type="retriever")
def traced_rag_retrieve(query: str, site_id: int):
    return rag.retrieve(query, site_id)


@traceable(name="text_run", run_type="chain")
def traced_full_text_run(query: str, site_id: int, language_code: str):
    return traced_text_run(query, site_id, language_code)


@traceable(name="tts_synthesize", run_type="tool")
def traced_tts_synthesize(sentence: str, tts_language_code: str):
    return tts.synthesize(sentence, tts_language_code)


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
    while True:
        msg = await websocket.receive()

        if msg.get("text") is not None:
            data = json.loads(msg["text"])
            if data.get("type") == "stop":
                timing["stop_received_at"] = time.perf_counter()
                await audio_q.put(None)
                return
            continue

        if msg.get("bytes") is not None:
            if timing.get("first_audio_at") is None:
                timing["first_audio_at"] = time.perf_counter()
            timing["last_audio_at"] = time.perf_counter()
            await audio_q.put(msg["bytes"])


async def send_tts_chunks(
    websocket: WebSocket,
    sentences: list[str],
    tts_language_code: str,
    trace_id: str,
    start_seq: int = 0,
) -> dict:
    """
    문장 리스트를 받아 문장 단위로 TTS 합성 후 순차 전송
    """
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


async def traced_llm_sentence_stream(
    query: str,
    contexts,
) -> AsyncIterator[str]:
    """
    llm.stream_generate_sentences 자체가 async generator라
    traceable로 직접 감싸기 애매해서 여기서 ls_trace로 묶음
    """
    with ls_trace(
        name="llm_sentence_stream",
        run_type="llm",
        inputs={
            "query": query,
            "context_count": len(contexts) if contexts is not None else 0,
        },
    ) as run:
        collected = []
        async for sentence in llm.stream_generate_sentences(query, contexts):
            if sentence and sentence.strip():
                collected.append(sentence.strip())
                yield sentence.strip()

        # outputs를 명시적으로 남기고 싶으면 이렇게
        if run is not None:
            try:
                run.end(outputs={"answer": " ".join(collected)})
            except Exception:
                pass


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

        with ls_trace(
            name="ws_voice_pipeline",
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
            },
        ):
            audio_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
            timing = {
                "ws_start_at": time.perf_counter(),
                "first_audio_at": None,
                "last_audio_at": None,
                "stop_received_at": None,
                "stt_final_at": None,
                "llm_start_at": None,
                "llm_first_sentence_at": None,
                "tts_first_audio_at": None,
            }

            async def _receiver_with_timer():
                await audio_receiver(websocket, audio_q, timing)

            recv_task = asyncio.create_task(_receiver_with_timer())

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "stt_start",
                "trace_id": trace_id,
            }, ensure_ascii=False))

            last_interim = ""
            last_final = ""
            last_lang2 = "ko"

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
                    last_lang2 = (ev.language_code or last_lang2).split("-")[0].lower()

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

            tts_language_code = map_tts_language(last_lang2)

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "llm_start",
                "trace_id": trace_id,
            }, ensure_ascii=False))

            # 1) 전체 답변 한 번에 생성
            if not realtime:
                timing["llm_start_at"] = time.perf_counter()

                result = await asyncio.to_thread(
                    traced_full_text_run,
                    query,
                    site_id,
                    last_lang2,
                )

                llm_total_ms = ms_delta(timing["llm_start_at"], time.perf_counter())
                answer_text = result.answer or ""

                await websocket.send_text(json.dumps({
                    "type": "final_text",
                    "site_id": site_id,
                    "language_code": last_lang2,
                    "query": result.query,
                    "answer": answer_text,
                    "trace_id": trace_id,
                }, ensure_ascii=False))

                tts_first_audio_latency_ms = None
                tts_total_ms = None
                tts_sentence_count = 0

                if tts_stream and answer_text.strip():
                    sentences = [s.strip() for s in answer_text.split("\n") if s.strip()]
                    if not sentences:
                        sentences = [answer_text]

                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "stage": "tts_start",
                        "trace_id": trace_id,
                    }, ensure_ascii=False))

                    tts_result = await send_tts_chunks(
                        websocket=websocket,
                        sentences=sentences,
                        tts_language_code=tts_language_code,
                        trace_id=trace_id,
                        start_seq=0,
                    )

                    tts_total_ms = tts_result["total_tts_ms"]
                    tts_sentence_count = tts_result["sentence_count"]
                    tts_first_audio_latency_ms = ms_delta(
                        timing["llm_start_at"],
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
                    llm_first_sentence_ms=None,
                    llm_total_ms=llm_total_ms,
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
                return

            # 2) 턴 종료 후 LLM sentence stream
            contexts = await asyncio.to_thread(traced_rag_retrieve, query, site_id)

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "llm_stream_start",
                "trace_id": trace_id,
            }, ensure_ascii=False))

            full_answer_parts: list[str] = []

            timing["llm_start_at"] = time.perf_counter()
            tts_started = False
            tts_total_ms = 0
            tts_sentence_count = 0
            tts_seq = 0

            async for sentence in traced_llm_sentence_stream(query, contexts):
                if not sentence.strip():
                    continue

                if timing["llm_first_sentence_at"] is None:
                    timing["llm_first_sentence_at"] = time.perf_counter()

                full_answer_parts.append(sentence)

                await websocket.send_text(json.dumps({
                    "type": "llm_sentence",
                    "text": sentence,
                    "language_code": last_lang2,
                    "trace_id": trace_id,
                }, ensure_ascii=False))

                if tts_stream:
                    if not tts_started:
                        tts_started = True
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
                        start_seq=tts_seq,
                    )

                    tts_seq = tts_result["next_seq"]
                    tts_total_ms += tts_result["total_tts_ms"]
                    tts_sentence_count += tts_result["sentence_count"]

                    if timing["tts_first_audio_at"] is None:
                        timing["tts_first_audio_at"] = tts_result["first_tts_audio_at"]

            final_answer = " ".join(full_answer_parts).strip()

            llm_first_sentence_ms = ms_delta(
                timing["llm_start_at"],
                timing["llm_first_sentence_at"],
            )
            total_llm_ms = ms_delta(
                timing["llm_start_at"],
                time.perf_counter(),
            )

            tts_first_audio_latency_ms = None
            if tts_stream and tts_started:
                tts_first_audio_latency_ms = ms_delta(
                    timing["llm_start_at"],
                    timing["tts_first_audio_at"],
                )

                await websocket.send_text(json.dumps({
                    "type": "status",
                    "stage": "tts_done",
                    "trace_id": trace_id,
                }, ensure_ascii=False))

            emit_latency_summary(
                trace_id=trace_id,
                site_id=site_id,
                realtime=True,
                tts_stream=tts_stream,
                stt_latency_ms=stt_latency_ms,
                llm_first_sentence_ms=llm_first_sentence_ms,
                llm_total_ms=total_llm_ms,
                tts_first_audio_latency_ms=tts_first_audio_latency_ms,
                tts_total_ms=tts_total_ms if tts_stream else None,
                tts_sentence_count=tts_sentence_count if tts_stream else 0,
                language_code=tts_language_code,
            )

            await websocket.send_text(json.dumps({
                "type": "final_text",
                "site_id": site_id,
                "language_code": last_lang2,
                "query": query,
                "answer": final_answer,
                "trace_id": trace_id,
            }, ensure_ascii=False))

            await websocket.send_text(json.dumps({
                "type": "status",
                "stage": "llm_stream_done",
                "trace_id": trace_id,
            }, ensure_ascii=False))

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