from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from openai import APIConnectionError, APITimeoutError, APIError, RateLimitError

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState
from app.graph.nodes.utils import (
    LANG_NAMES,
    append_trace_flow,
    build_messages,
    build_persona_block,
    get_language,
)

CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8080")
_SEARCH_THRESHOLD = 0.4   # 유사도 미달 시 nearby_places fallback
_ALLOWED_EMOTIONS = {"GUIDING", "HAPPY", "THINKING", "SORRY", "EXCITED"}


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────

def _kakao_walk_url(
    origin_name: str, origin_lat: float, origin_lng: float,
    dest_name: str, dest_lat: float, dest_lng: float,
) -> str:
    """카카오맵 도보 길찾기 URL 생성 (출발지 → 목적지). Unity WebView에서 열면 됨."""
    enc_origin = quote(origin_name)
    enc_dest = quote(dest_name)
    return (
        f"https://map.kakao.com/link/by/walk/"
        f"{enc_origin},{origin_lat},{origin_lng}/"
        f"{enc_dest},{dest_lat},{dest_lng}"
    )


def _search_place(site_id: int, query: str) -> Optional[Dict[str, Any]]:
    """Core /places/search 호출. threshold 미달이면 None."""
    try:
        resp = requests.get(
            f"{CORE_BASE_URL}/internal/v1/places/search",
            params={"siteId": site_id, "q": query, "threshold": _SEARCH_THRESHOLD},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"[navigation] places/search 오류: {type(e).__name__}")
        return None


# ── 노드 팩토리 ───────────────────────────────────────────────────────────────

def make_navigation_node(llm: OpenAILLM):
    """장소 안내 노드 팩토리.

    struct_db_node를 대체한다.

    동작:
    1. place_name_query 있음 → Core /places/search → 장소 찾으면 map_url 생성 + LLM 답변
    2. 유사도 미달(None 반환) → nearby_places 기반 fallback (기존 struct_db 동작)
    """

    def navigation_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language = get_language(state)
        lang_name = LANG_NAMES.get(user_language, user_language.upper())
        site_id: Optional[int] = state.get("site_id")
        device_location: Dict[str, Any] = state.get("device_location") or {}
        nearby_places: List[Dict[str, Any]] = state.get("nearby_places") or []
        system_prompt: str = state.get("system_prompt") or ""

        trace = append_trace_flow(state, "navigation")

        # 마스코트 페르소나
        style = (
            state.get("mascot_struct_db_style")
            or state.get("mascot_base_persona")
            or ""
        )
        persona_block = build_persona_block(
            base_prompt=system_prompt,
            style=style,
            user_language=user_language,
            lang_name=lang_name,
            name=state.get("mascot_name") or "",
            ko_fallback="당신은 관광지의 친절한 위치 안내원입니다.",
            foreign_fallback="You are a friendly tourist guide assistant.",
        )

        # ── 1. 특정 장소 검색 ──────────────────────────────────────────────
        # place_name_query가 있을 때만 places/search 호출
        # (카테고리 질문: "화장실 어디야?" → place_name_query=null → 바로 nearby_places fallback)
        place_name_query: Optional[str] = state.get("place_name_query")
        place: Optional[Dict[str, Any]] = None
        if site_id and place_name_query:
            place = _search_place(site_id, place_name_query)

        if place:
            # ── 1-a. 카카오맵 도보 URL 생성 ───────────────────────────────
            place_name = place.get("name", "")
            dest_lat = place.get("latitude")
            dest_lng = place.get("longitude")
            origin_lat = device_location.get("latitude")
            origin_lng = device_location.get("longitude")

            map_url: Optional[str] = None

            if all(v is not None for v in [origin_lat, origin_lng, dest_lat, dest_lng]):
                map_url = _kakao_walk_url(
                    "현재위치", origin_lat, origin_lng,
                    place_name, dest_lat, dest_lng,
                )
            elif dest_lat is not None and dest_lng is not None:
                enc_dest = quote(place_name)
                map_url = f"https://map.kakao.com/link/to/{enc_dest},{dest_lat},{dest_lng}"

            # ── 1-b. LLM 답변 생성 ────────────────────────────────────────
            if user_language == "ko":
                system_msg = (
                    f"{persona_block}\n"
                    "규칙:\n"
                    "  - 아래 장소 정보를 바탕으로 1~2문장으로 위치를 안내하세요\n"
                    "  - 이모지나 특수문자는 사용하지 마세요\n\n"
                    f"장소: {place_name} (카테고리: {place.get('category')})\n\n"
                    "반드시 아래 JSON 형식으로만 응답하세요 (마크다운 없이):\n"
                    '{"answer": "자연어 답변", "emotion": "GUIDING"}'
                )
            else:
                system_msg = (
                    f"{persona_block}\n"
                    "Rules:\n"
                    f"  - CRITICAL: Your entire answer MUST be in {lang_name}.\n"
                    f"  - Respond in {lang_name}, 1-2 sentences\n"
                    "  - No emoji, no special characters\n\n"
                    f"Place: {place_name} (category: {place.get('category')})\n\n"
                    "Return ONLY valid JSON (no markdown):\n"
                    '{"answer": "...", "emotion": "GUIDING"}'
                )

            messages = build_messages(state, system_msg, text)
            method = "llm_navigation"
            answer_text = ""
            emotion = "GUIDING"
            error_msg = ""
            raw = ""

            try:
                raw = llm.chat(messages, max_tokens=200)
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response must be a JSON object")
                answer_text = parsed.get("answer", "") if isinstance(parsed.get("answer"), str) else ""
                raw_emotion = parsed.get("emotion", "GUIDING")
                emotion = raw_emotion if raw_emotion in _ALLOWED_EMOTIONS else "GUIDING"
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                answer_text = raw
                method = "llm_raw_fallback"
                error_msg = str(e)
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
                method = "llm_api_error"
                error_msg = str(e)

            if answer_text:
                trace["navigation"] = {
                    "status": "ok",
                    "method": method,
                    "place": place_name,
                    "similarity": place.get("similarity"),
                    "map_url": map_url,
                }
                if error_msg:
                    trace["navigation"]["error"] = error_msg
                return {
                    "answer_text": answer_text,
                    "place_id": place.get("placeId"),
                    "map_url": map_url,
                    "emotion": emotion,
                    "category": "DIRECTION",
                    "check_result": "good",
                    "trace": trace,
                }

            # LLM 실패 → nearby_places fallback 으로 계속 진행
            trace["navigation"] = {"status": "llm_failed_fallback", "error": error_msg}

        # ── 2. Fallback: nearby_places 기반 (기존 struct_db 동작) ─────────
        if not nearby_places:
            trace["navigation"] = {
                "status": "no_context",
                "place_name_query": place_name_query,
                "query": text,
            }
            return {"answer_text": "", "check_result": "bad", "trace": trace}

        places_lines = []
        for p in nearby_places:
            same = "✓ 같은 구역" if p.get("sameZone") else "다른 구역"
            dist = p.get("distanceM")
            dist_str = f"{dist:.0f}m" if isinstance(dist, (int, float)) else "거리 미상"
            desc = p.get("description") or ""
            places_lines.append(
                f"- placeId={p.get('placeId')} [{p.get('name')}] "
                f"카테고리: {p.get('category')} | {same} | {dist_str} | {desc}"
            )
        places_text = "\n".join(places_lines)

        if user_language == "ko":
            system_msg = (
                f"{persona_block}\n"
                "규칙:\n"
                "  - 아래 장소 목록만 참고하세요 (장소를 지어내지 마세요)\n"
                "  - 같은 구역(✓ 같은 구역) 장소를 우선, 가까운 순서로 안내하세요\n"
                "  - 장소 이름과 거리를 포함해 1~2문장으로 답하세요\n"
                "  - 이모지나 특수문자는 사용하지 마세요\n\n"
                f"주변 장소 목록:\n{places_text}\n\n"
                "반드시 아래 JSON 형식으로만 응답하세요 (마크다운 없이):\n"
                '{"answer": "자연어 답변", "place_id": <placeId 또는 null>, '
                '"emotion": "GUIDING", "category": "DIRECTION"}'
            )
        else:
            system_msg = (
                f"{persona_block}\n"
                "Rules:\n"
                f"  - CRITICAL: Your entire answer MUST be in {lang_name}. Do NOT include any Korean words.\n"
                f"  - Respond in {lang_name}, 1-2 sentences\n"
                "  - Use ONLY the provided nearby places list (do not invent places)\n"
                "  - Prioritize same-zone (✓ 같은 구역) places and shorter distances\n"
                "  - Mention the place name and distance\n"
                "  - No emoji, no special characters\n\n"
                f"Nearby places:\n{places_text}\n\n"
                "Return ONLY valid JSON (no markdown):\n"
                '{"answer": "...", "place_id": <placeId or null>, '
                '"emotion": "GUIDING", "category": "DIRECTION"}'
            )

        messages = build_messages(state, system_msg, text)
        method = "llm_fallback_nearby"
        answer_text = ""
        place_id = None
        emotion = "GUIDING"
        error_msg = ""
        raw = ""

        try:
            raw = llm.chat(messages, max_tokens=200)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response must be a JSON object")
            answer_text = parsed.get("answer", "") if isinstance(parsed.get("answer"), str) else ""
            raw_place_id = parsed.get("place_id")
            if isinstance(raw_place_id, bool) or raw_place_id is None:
                place_id = None
            elif isinstance(raw_place_id, int):
                place_id = raw_place_id
            elif isinstance(raw_place_id, float) and raw_place_id.is_integer():
                place_id = int(raw_place_id)
            raw_emotion = parsed.get("emotion", "GUIDING")
            emotion = raw_emotion if raw_emotion in _ALLOWED_EMOTIONS else "GUIDING"
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            answer_text = raw
            method = "llm_raw_fallback"
            error_msg = str(e)
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            method = "llm_api_error"
            error_msg = str(e)

        if not answer_text:
            trace["navigation"] = {"status": "llm_failed", "method": method, "error": error_msg}
            return {"answer_text": "", "check_result": "bad", "trace": trace}

        # LLM이 선택한 place_id로 nearby_places에서 좌표 찾아 map_url 생성
        map_url: Optional[str] = None
        if place_id is not None:
            selected = next((p for p in nearby_places if p.get("placeId") == place_id), None)
            if selected:
                dest_lat = selected.get("latitude")
                dest_lng = selected.get("longitude")
                origin_lat = device_location.get("latitude")
                origin_lng = device_location.get("longitude")
                dest_name = selected.get("name", "")
                if all(v is not None for v in [origin_lat, origin_lng, dest_lat, dest_lng]):
                    map_url = _kakao_walk_url(
                        "현재위치", origin_lat, origin_lng,
                        dest_name, dest_lat, dest_lng,
                    )
                elif dest_lat is not None and dest_lng is not None:
                    map_url = f"https://map.kakao.com/link/to/{quote(dest_name)},{dest_lat},{dest_lng}"

        trace["navigation"] = {
            "status": "fallback_nearby",
            "method": method,
            "place_id": place_id,
            "places_count": len(nearby_places),
            "map_url": map_url,
        }
        if error_msg:
            trace["navigation"]["error"] = error_msg

        return {
            "answer_text": answer_text,
            "place_id": place_id,
            "map_url": map_url,
            "emotion": emotion,
            "category": "DIRECTION",
            "check_result": "good",
            "trace": trace,
        }

    return navigation_node
