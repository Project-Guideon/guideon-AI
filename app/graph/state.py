from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    # ── 입력 ──────────────────────────────────────────────────────────────
    site_id: int               # 어느 관광지(사이트)냐
    audio: bytes               # 원본 오디오 bytes

    # ── STT 출력 ──────────────────────────────────────────────────────────
    transcript: str            # STT 변환 텍스트
    language_code: str         # 감지 언어 2자리: "ko" | "en" | "zh" | "ja" 등
    user_language: str         # 원언어 보존 (language_code 와 동일값; 끝까지 유지)

    # ── 정규화 ────────────────────────────────────────────────────────────
    normalized_text: str       # 잡음·공백 정제 후 텍스트

    # ── 의도 분류 ─────────────────────────────────────────────────────────
    intent: Literal["smalltalk", "info_request"]

    # ── 정보 유형 분류 ────────────────────────────────────────────────────
    info_type: Literal["rag", "map_tool", "struct_db", "direct_llm"]

    # ── RAG 파이프라인 ────────────────────────────────────────────────────
    retrieval_query_ko: str            # 검색용 한국어 쿼리
    top_k: int                         # 초기 5, 재시도 시 10
    retry_count: int                   # 현재 재시도 횟수 (최대 2)
    retrieved_chunks: List[Dict[str, Any]]   # 검색 결과 청크 리스트
    # 각 chunk dict: {chunk_id, doc_id, content, metadata, similarity}

    # ── 답변 ──────────────────────────────────────────────────────────────
    answer_text: str                   # LLM 생성 최종 답변
    check_result: Literal["good", "retry", "bad"]  # 답변 품질 판정

    # ── TTS ───────────────────────────────────────────────────────────────
    tts_text: str              # TTS용 보정 텍스트
    tts_audio: bytes           # Google TTS 반환 오디오 bytes

    # ── 디버깅 ────────────────────────────────────────────────────────────
    trace: Dict[str, Any]      # 노드별 로그/메타 기록
