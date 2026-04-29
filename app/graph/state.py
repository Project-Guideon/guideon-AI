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

    # ── 의도 분류 (단일 gate → 4분기 + 순위 기반 fallback) ────────────────
    intent_ranking: List[str]      # ["rag", "event", "struct_db", "smalltalk"] 순위
    current_intent_index: int      # 현재 시도 중인 의도 인덱스 (0부터)
    fallback_next: str             # fallback_dispatch가 설정하는 라우팅 키

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

    # ── 대화 내역 (Redis) ─────────────────────────────────────────────────
    chat_history: List[Dict[str, Any]]  # [{"role": "user"|"assistant", "content": "..."}]

    # ── 요청 디바이스 / 마스코트 정보 ────────────────────────────────────
    device_id: str                  # 디바이스 ID (fetch_places_node가 Spring Boot places API 호출 시 사용)
    device_location: Dict[str, Any]  # optional {latitude, longitude}; currently kept for future use
    system_prompt: str              # 마스코트 캐릭터 프롬프트 (tb_mascot.system_prompt)
    mascot_name: str                # 마스코트 이름 (예: "가온이")
    mascot_greeting: str            # 인사말 (smalltalk 참고용)
    mascot_base_persona: str        # 공통 fallback 페르소나 (각 style 비어있을 때 사용)
    mascot_smalltalk_style: str     # 일상대화 말투 지침
    mascot_struct_db_style: str     # 위치 안내 스타일
    mascot_RAG_style: str           # RAG 답변 스타일
    mascot_event_style: str         # 이벤트/운영정보 스타일

    # ── intent_gate 추출 결과 ────────────────────────────────────────────
    place_category: Optional[str]   # struct_db 라우트일 때 추출된 장소 카테고리 (ex: RESTROOM, PARKING)

    # ── Spring Boot Core 에서 전달받은 위치 context ──────────────────────
    nearby_places: List[Dict[str, Any]]  # [{placeId, name, category, description, distanceM, sameZone}]
    daily_infos: List[Dict[str, Any]]    # [{placeName, infoType, content}]

    # ── struct_db / answer 결과 메타 ─────────────────────────────────────
    place_id: Optional[int]    # 언급된 장소 ID (display hint용)
    emotion: str               # 캐릭터 감정: GUIDING | HAPPY | THINKING | SORRY | EXCITED
    category: str              # 질문 유형: DIRECTION | HOURS | FACILITY | HISTORY | GENERAL | ERROR

    # ── 디버깅 ────────────────────────────────────────────────────────────
    trace: Dict[str, Any]      # 노드별 로그/메타 기록
