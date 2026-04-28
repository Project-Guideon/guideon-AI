"""
WebSocket 스트리밍 엔드포인트
- WS /ws/stream : 오디오 스트림 수신 → STT → LangGraph 파이프라인 → TTS 스트리밍 응답

WebSocket 라이프사이클 원칙:
- `connected` 플래그로 연결 상태를 단일 진실 공급원으로 관리
- 모든 send는 safe_send()를 통해서만 수행 → 연결 끊긴 뒤 send 원천 차단
- disconnect 발생 즉시 connected=False, 이후 모든 루프 조기 종료
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.api_core import exceptions as google_exceptions
from langsmith import trace as ls_trace, traceable

from app.core.dependencies import stt, tts, traced_text_run
from app.core.services.chat_history import load_chat_history

router = APIRouter()


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def _new_trace_id() -> str:
    return str(uuid.uuid4())


def map_tts_language(lang2: str) -> str:
    lang2 = (lang2 or "ko").split("-")[0].lower()
    mapping = {"ko": "ko-KR", "en": "en-US", "ja": "ja-JP", "zh": "cmn-CN"}
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


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _first_non_empty(*values: object) -> object:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def _prompt_config_value(raw_mascot: dict, prompt_config: dict, flat_key: str, chat_key: str) -> object:
    return _first_non_empty(raw_mascot.get(flat_key), prompt_config.get(chat_key), prompt_config.get(flat_key))


def normalize_mascot_payload(raw_mascot: object, raw_start: object | None = None) -> dict:
    raw_mascot = _as_dict(raw_mascot)
    raw_start = _as_dict(raw_start)
    prompt_config = _as_dict(raw_start.get("promptConfig") or raw_mascot.get("promptConfig"))
    return {
        "system_prompt":          _first_non_empty(raw_mascot.get("system_prompt"), raw_start.get("systemPrompt")),
        "mascot_name":            _first_non_empty(raw_mascot.get("mascot_name"), raw_start.get("name")),
        "mascot_greeting":        _first_non_empty(raw_mascot.get("mascot_greeting"), raw_start.get("greetingMsg")),
        "mascot_base_persona":    _prompt_config_value(raw_mascot, prompt_config, "mascot_base_persona", "base_persona"),
        "mascot_smalltalk_style": _prompt_config_value(raw_mascot, prompt_config, "mascot_smalltalk_style", "smalltalk_style"),
        "mascot_struct_db_style": _prompt_config_value(raw_mascot, prompt_config, "mascot_struct_db_style", "struct_db_style"),
        "mascot_RAG_style":       _prompt_config_value(raw_mascot, prompt_config, "mascot_RAG_style", "RAG_style"),
        "mascot_event_style":     _prompt_config_value(raw_mascot, prompt_config, "mascot_event_style", "event_node_style"),
    }


def normalize_daily_infos_payload(raw_context: object) -> list[dict]:
    if not isinstance(raw_context, dict):
        return []
    raw_daily_infos = raw_context.get("dailyInfos") or []
    if not isinstance(raw_daily_infos, list):
        return []

    daily_infos: list[dict] = []
    for item in raw_daily_infos:
        if not isinstance(item, dict):
            continue
        daily_infos.append({
            "placeName": item.get("placeName", "") or "",
            "infoType": item.get("infoType", "") or "",
            "content": item.get("content", "") or "",
        })
    return daily_infos


def normalize_device_location_payload(raw_location: object) -> dict:
    if not isinstance(raw_location, dict):
        return {}
    latitude = raw_location.get("latitude")
    longitude = raw_location.get("longitude")
    if latitude is None or longitude is None:
        return {}
    return {"latitude": latitude, "longitude": longitude}


def extract_answer_text(result) -> str:
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


# ──────────────────────────────────────────────
# Traceable 래퍼
# ──────────────────────────────────────────────

@traceable(name="emit_latency_summary", run_type="tool")
def emit_latency_summary(
    trace_id: str, site_id: int, realtime: bool, tts_stream: bool,
    stt_latency_ms: int | None, qa_first_sentence_ms: int | None,
    qa_total_ms: int | None, tts_first_audio_latency_ms: int | None,
    tts_total_ms: int | None, tts_sentence_count: int, language_code: str,
):
    return {
        "trace_id": trace_id, "site_id": site_id,
        "realtime": realtime, "tts_stream": tts_stream,
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
    device_id: str | None = None,
    chat_history: list[dict] | None = None,
    daily_infos: list[dict] | None = None,
    device_location: dict | None = None,
):
    mascot = normalize_mascot_payload(mascot)
    return traced_text_run(
        query=query,
        site_id=site_id,
        language_code=language_code,
        mascot=mascot,
        device_id=device_id,
        chat_history=chat_history,
        daily_infos=daily_infos,
        device_location=device_location,
    )


# ──────────────────────────────────────────────
# audio_receiver
# ──────────────────────────────────────────────

async def audio_receiver(
    websocket: WebSocket,
    audio_q: "asyncio.Queue[Optional[bytes]]",
    timing: dict,
):
    """
    클라이언트 → 서버 방향 수신 루프.

    종료 조건:
    - {"type": "stop"} 텍스트 프레임 수신
    - WebSocketDisconnect 예외
    - websocket.disconnect 메시지 수신
    - receive() 호출 후 RuntimeError (이미 disconnect된 소켓)

    어떤 경로로 종료되든 finally에서 audio_q.put(None) 보장.
    """
    try:
        while True:
            try:
                msg = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                # WebSocketDisconnect: Starlette 예외 경로
                # RuntimeError: disconnect 후 receive() 재호출 방어
                break

            # Starlette가 disconnect를 예외 없이 dict로 반환하는 경우
            if msg.get("type") == "websocket.disconnect":
                break

            # 텍스트 프레임: control message
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except (json.JSONDecodeError, ValueError):
                    logger.warning("audio_receiver: invalid JSON — ignoring")
                    continue

                if data.get("type") == "stop":
                    timing["stop_received_at"] = time.perf_counter()
                    return  # finally가 None을 넣어줌

                continue

            # 바이너리 프레임: PCM audio chunk
            if msg.get("bytes") is not None:
                if timing.get("first_audio_at") is None:
                    timing["first_audio_at"] = time.perf_counter()
                timing["last_audio_at"] = time.perf_counter()
                await audio_q.put(msg["bytes"])

    finally:
        # 어떤 경로로 종료되든 STT 스트림에 종료 신호 전달
        await audio_q.put(None)


# ──────────────────────────────────────────────
# TTS chunk 전송
# ──────────────────────────────────────────────

async def send_tts_chunks(
    sentences: list[str],
    tts_language_code: str,
    trace_id: str,
    safe_send,           # ws_stream 스코프의 safe_send 함수
    start_seq: int = 0,
    mark_final: bool = True,
) -> dict:
    non_empty = [s.strip() for s in sentences if s and s.strip()]
    first_tts_audio_at = None
    total_tts_ms = 0
    current_seq = start_seq

    for idx, sentence in enumerate(non_empty):
        t0 = time.perf_counter()
        audio_chunk = await asyncio.to_thread(traced_tts_synthesize, sentence, tts_language_code)
        total_tts_ms += round((time.perf_counter() - t0) * 1000)

        if first_tts_audio_at is None:
            first_tts_audio_at = time.perf_counter()

        sent = await safe_send({
            "type": "tts_chunk",
            "seq": current_seq,
            "text": sentence,
            "audio_format": "mp3",
            "audio_b64": base64.b64encode(audio_chunk).decode("utf-8"),
            "language_code": tts_language_code,
            "is_final": mark_final and idx == len(non_empty) - 1,
            "trace_id": trace_id,
        })
        if not sent:
            break  # 클라이언트 disconnect → 이후 전송 중단

        current_seq += 1

    return {
        "first_tts_audio_at": first_tts_audio_at,
        "total_tts_ms": total_tts_ms,
        "sentence_count": len(non_empty),
        "next_seq": current_seq,
    }


# ──────────────────────────────────────────────
# WebSocket 핸들러
# ──────────────────────────────────────────────

@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    trace_id = _new_trace_id()

    # ── 연결 상태 플래그 ──────────────────────────
    # True인 동안만 send 허용. disconnect/에러 발생 시 즉시 False로 변경.
    connected = True

    async def safe_send(payload: dict) -> bool:
        """
        WebSocket send 단일 창구.
        - connected=False 이면 즉시 False 반환 (send 시도 안 함)
        - 전송 실패(disconnect/RuntimeError) 시 connected=False 설정 후 False 반환
        """
        nonlocal connected
        if not connected:
            return False
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            return True
        except (WebSocketDisconnect, RuntimeError):
            connected = False
            return False

    recv_task: Optional[asyncio.Task] = None

    try:
        # ── start 메시지 수신 ─────────────────────
        try:
            raw_start = await websocket.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            return

        try:
            start = json.loads(raw_start)
            if not isinstance(start, dict):
                await safe_send({
                    "type": "error", "code": "BAD_REQUEST",
                    "message": "start payload must be a JSON object",
                    "trace_id": trace_id,
                })
                return
            if start.get("type") != "start":
                await safe_send({
                    "type": "error", "code": "BAD_REQUEST",
                    "message": "first message must be {type:'start'}",
                    "trace_id": trace_id,
                })
                return
            site_id        = int(start.get("siteId") or start.get("site_id") or 1)
            stt_language   = start.get("language") or start.get("language_code") or "ko-KR"
            sample_rate_hz = int(start.get("sampleRateHz") or start.get("sample_rate_hz") or 16000)
            device_id_raw  = start.get("deviceId") or start.get("device_id")
            device_id      = str(device_id_raw).strip() if device_id_raw is not None else None
            device_id      = device_id or None
            session_id_raw = start.get("sessionId") or start.get("session_id")
            session_id     = str(session_id_raw).strip() if session_id_raw is not None else None
            session_id     = session_id or None
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            await safe_send({
                "type": "error", "code": "BAD_REQUEST",
                "message": f"invalid start payload: {exc}",
                "trace_id": trace_id,
            })
            return
        interim_results = bool(start["interimResults"] if "interimResults" in start else start.get("interim_results", True))
        tts_stream_on  = bool(start["ttsStream"] if "ttsStream" in start else start.get("tts_stream", True))
        realtime       = bool(start.get("realtime", False))
        mascot         = normalize_mascot_payload(start.get("mascot"), start)
        daily_infos    = normalize_daily_infos_payload(start.get("context"))
        device_location = normalize_device_location_payload(start.get("deviceLocation") or start.get("device_location"))
        chat_history   = await load_chat_history(session_id) if session_id else []


        with ls_trace(
            name="ws_voice_textqa_pipeline",
            run_type="chain",
            metadata={"trace_id": trace_id, "site_id": site_id,
                      "session_id": session_id,
                      "device_id": device_id,
                      "daily_info_count": len(daily_infos),
                      "has_device_location": bool(device_location),
                      "tts_stream": tts_stream_on, "realtime": realtime},
            inputs={"site_id": site_id, "stt_language": stt_language,
                    "sample_rate_hz": sample_rate_hz, "interim_results": interim_results,
                    "tts_stream": tts_stream_on, "realtime": realtime,
                    "session_id": session_id, "device_id": device_id,
                    "device_location": device_location,
                    "daily_info_count": len(daily_infos),
                    "chat_history_count": len(chat_history),
                    "mascot": mascot},
        ):
            audio_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=250)
            timing = {
                "ws_start_at": time.perf_counter(),
                "first_audio_at": None, "last_audio_at": None,
                "stop_received_at": None, "stt_final_at": None,
                "qa_start_at": None, "qa_first_sentence_at": None,
                "tts_first_audio_at": None,
            }

            # audio_receiver는 백그라운드에서 오디오를 수신해 audio_q에 적재
            recv_task = asyncio.create_task(audio_receiver(websocket, audio_q, timing))

            # ── STT 스트리밍 ──────────────────────
            await safe_send({"type": "status", "stage": "stt_start", "trace_id": trace_id})

            final_parts: list[str] = []
            current_utterance_interim = ""
            unfinalised_interims: list[str] = []
            last_lang_code = stt_language

            with ls_trace(
                name="stt_stream", run_type="tool",
                metadata={"trace_id": trace_id, "primary_language": stt_language,
                          "sample_rate_hz": sample_rate_hz, "interim_results": interim_results},
            ):
                try:
                    async for ev in stt.stream_events(
                        audio_q,
                        primary_language=stt_language,
                        sample_rate_hz=sample_rate_hz,
                        interim_results=interim_results,
                        single_utterance=False,
                    ):
                        # disconnect 이후에는 STT 이벤트 처리 중단
                        if not connected:
                            break

                        last_lang_code = ev.language_code or last_lang_code

                        if ev.is_final:
                            if (current_utterance_interim
                                    and current_utterance_interim.strip() not in ev.transcript):
                                unfinalised_interims.append(current_utterance_interim.strip())
                            current_utterance_interim = ""
                            final_parts.append(ev.transcript)
                        else:
                            current_utterance_interim = ev.transcript

                        await safe_send({
                            "type": "stt_final" if ev.is_final else "stt_interim",
                            "text": ev.transcript,
                            "language_code": ev.language_code,
                            "confidence": ev.confidence,
                            "is_final": ev.is_final,
                            "trace_id": trace_id,
                        })

                except google_exceptions.Aborted:
                    # Google STT 스트림 timeout — 연결 문제로 처리하지 않고 조용히 종료
                    logger.warning("stt stream aborted (timeout) trace_id=%s", trace_id)
                except Exception:
                    logger.exception("stt stream error trace_id=%s", trace_id)

            timing["stt_final_at"] = time.perf_counter()

            # recv_task 완료 대기 (audio_receiver가 None을 넣은 뒤 종료)
            if recv_task and not recv_task.done():
                try:
                    await asyncio.wait_for(recv_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    recv_task.cancel()

            # disconnect 됐으면 이후 파이프라인 실행 불필요
            if not connected:
                return

            stt_latency_ms = ms_delta(timing["stop_received_at"], timing["stt_final_at"])
            await safe_send({"type": "status", "stage": "stt_done", "trace_id": trace_id})

            # ── transcript 조합 ───────────────────
            sep = "" if last_lang_code.startswith(("ja", "zh")) else " "
            finals_text = sep.join(final_parts).strip()

            if current_utterance_interim.strip() and current_utterance_interim.strip() not in finals_text:
                unfinalised_interims.append(current_utterance_interim.strip())

            prefix = sep.join(p for p in unfinalised_interims if p and p not in finals_text).strip()
            query = (
                (prefix + sep + finals_text).strip() if prefix and finals_text
                else prefix or finals_text or current_utterance_interim.strip()
            )

            if not query:
                await safe_send({
                    "type": "error", "code": "EMPTY_TRANSCRIPT",
                    "message": "no transcript", "trace_id": trace_id,
                })
                return

            tts_language_code = map_tts_language(last_lang_code)

            # ── LangGraph QA ──────────────────────
            await safe_send({"type": "status", "stage": "graph_start", "trace_id": trace_id})
            timing["qa_start_at"] = time.perf_counter()

            qa_result = await asyncio.to_thread(
                traced_ws_text_qa_invoke,
                query,
                site_id,
                last_lang_code,
                mascot,
                device_id,
                chat_history,
                daily_infos,
                device_location,
            )
            qa_total_ms = ms_delta(timing["qa_start_at"], time.perf_counter())

            answer_text = extract_answer_text(qa_result) or "죄송해요. 답변을 생성하지 못했어요."
            sentences = split_sentences(answer_text)
            qa_category = (qa_result.get("category") if isinstance(qa_result, dict) else getattr(qa_result, "category", None)) or "GENERAL"

            tts_first_audio_latency_ms = None
            tts_total_ms = None
            tts_sentence_count = 0

            # ── 응답 전송 (realtime / non-realtime) ─
            if realtime:
                await safe_send({"type": "status", "stage": "graph_stream_start", "trace_id": trace_id})

                for idx, sentence in enumerate(sentences):
                    if not connected:
                        break

                    if timing["qa_first_sentence_at"] is None:
                        timing["qa_first_sentence_at"] = time.perf_counter()

                    await safe_send({"type": "llm_sentence", "text": sentence,
                                     "language_code": last_lang_code, "trace_id": trace_id})

                    if tts_stream_on:
                        if idx == 0:
                            await safe_send({"type": "status", "stage": "tts_start", "trace_id": trace_id})

                        tts_result = await send_tts_chunks(
                            sentences=[sentence],
                            tts_language_code=tts_language_code,
                            trace_id=trace_id, safe_send=safe_send,
                            start_seq=idx, mark_final=False,
                        )
                        tts_total_ms = (tts_total_ms or 0) + tts_result["total_tts_ms"]
                        tts_sentence_count += tts_result["sentence_count"]
                        if timing["tts_first_audio_at"] is None:
                            timing["tts_first_audio_at"] = tts_result["first_tts_audio_at"]

                qa_first_sentence_ms = ms_delta(timing["qa_start_at"], timing["qa_first_sentence_at"])

                if tts_stream_on and timing["tts_first_audio_at"] is not None:
                    tts_first_audio_latency_ms = ms_delta(timing["qa_start_at"], timing["tts_first_audio_at"])
                    await safe_send({"type": "status", "stage": "tts_done", "trace_id": trace_id})

                await safe_send({
                    "type": "final_text", "site_id": site_id,
                    "language_code": last_lang_code, "query": query,
                    "answer": answer_text, "category": qa_category, "trace_id": trace_id,
                })
                await safe_send({"type": "status", "stage": "graph_stream_done", "trace_id": trace_id})

                emit_latency_summary(
                    trace_id=trace_id, site_id=site_id, realtime=True,
                    tts_stream=tts_stream_on, stt_latency_ms=stt_latency_ms,
                    qa_first_sentence_ms=qa_first_sentence_ms, qa_total_ms=qa_total_ms,
                    tts_first_audio_latency_ms=tts_first_audio_latency_ms,
                    tts_total_ms=tts_total_ms if tts_stream_on else None,
                    tts_sentence_count=tts_sentence_count if tts_stream_on else 0,
                    language_code=tts_language_code,
                )

            else:
                await safe_send({
                    "type": "final_text", "site_id": site_id,
                    "language_code": last_lang_code, "query": query,
                    "answer": answer_text, "category": qa_category, "trace_id": trace_id,
                })

                qa_first_sentence_ms = None

                if tts_stream_on and answer_text.strip():
                    await safe_send({"type": "status", "stage": "tts_start", "trace_id": trace_id})

                    tts_result = await send_tts_chunks(
                        sentences=sentences or [answer_text],
                        tts_language_code=tts_language_code,
                        trace_id=trace_id, safe_send=safe_send, start_seq=0,
                    )
                    tts_total_ms = tts_result["total_tts_ms"]
                    tts_sentence_count = tts_result["sentence_count"]
                    tts_first_audio_latency_ms = ms_delta(timing["qa_start_at"], tts_result["first_tts_audio_at"])
                    await safe_send({"type": "status", "stage": "tts_done", "trace_id": trace_id})

                emit_latency_summary(
                    trace_id=trace_id, site_id=site_id, realtime=False,
                    tts_stream=tts_stream_on, stt_latency_ms=stt_latency_ms,
                    qa_first_sentence_ms=qa_first_sentence_ms, qa_total_ms=qa_total_ms,
                    tts_first_audio_latency_ms=tts_first_audio_latency_ms,
                    tts_total_ms=tts_total_ms, tts_sentence_count=tts_sentence_count,
                    language_code=tts_language_code,
                )

            # ── 정상 완료 ─────────────────────────
            await safe_send({"type": "done", "trace_id": trace_id})

    except WebSocketDisconnect:
        connected = False

    except Exception:
        logger.exception("ws_stream unhandled error trace_id=%s", trace_id)
        await safe_send({
            "type": "error", "code": "INTERNAL",
            "message": "내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "trace_id": trace_id,
        })

    finally:
        connected = False
        # recv_task가 아직 살아있으면 정리
        if recv_task and not recv_task.done():
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass
        # WebSocket close 시도 (이미 닫혔으면 무시)
        try:
            await websocket.close()
        except Exception:
            pass
