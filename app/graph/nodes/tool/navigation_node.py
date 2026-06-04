from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
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
_ANSWER_PLACEHOLDERS = frozenset({"위치 안내 문구만", "..."})
_MAP_GUIDE = {
    "ko": "자세한 길찾기는 화면의 지도를 보고 따라가면 됩니다.",
    "en": "For detailed directions, please follow the map on the screen.",
    "ja": "詳しい道案内は、画面の地図をご覧ください。",
    "zh": "详细路线请参考屏幕上的地图。",
}


def _extract_json_from_mixed(text: str) -> Optional[Dict[str, Any]]:
    """자연어+JSON 혼합 응답에서 JSON 객체만 추출한다."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _text_before_json(text: str) -> str:
    """JSON 블록 앞의 자연어 부분만 반환한다."""
    idx = text.find('{')
    return text[:idx].strip() if idx != -1 else text.strip()


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


def _match_from_nearby(
    query: str, nearby_places: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """이미 받아둔 nearby_places에서 장소명을 먼저 매칭한다.

    거리순으로 가져온 목록이므로, 여기서 찾으면 pg_trgm 사이트 전체 검색을 생략한다.
    정확/부분 문자열 매칭을 우선하고, 없으면 difflib 유사도가 threshold 이상인 항목을 채택한다.
    """
    if not nearby_places:
        return None

    # 공백 제거 후 빈 문자열이면 매칭 생략 (빈 query가 부분 매칭으로 오인되는 것 방지)
    norm_q = "".join(query.split()).lower()
    if not norm_q:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for p in nearby_places:
        name = p.get("name") or ""
        norm_name = "".join(name.split()).lower()
        if not norm_name:
            continue

        # 정확 매칭 또는 부분 문자열 포함 → 즉시 채택
        if norm_q == norm_name or norm_q in norm_name or norm_name in norm_q:
            return p

        score = SequenceMatcher(None, norm_q, norm_name).ratio()
        if score > best_score:
            best_score = score
            best = p

    if best is not None and best_score >= _SEARCH_THRESHOLD:
        return best
    return None


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
        # place_name_query가 있을 때만 검색 수행
        # (카테고리 질문: "화장실 어디야?" → place_name_query=null → 바로 nearby_places fallback)
        #
        # 검색 순서:
        #   1) 이미 받아둔 nearby_places(거리순)에서 이름 매칭 → 있으면 pg_trgm 생략
        #   2) 없으면 pg_trgm으로 사이트 전체 검색 (멀리 있는 장소 대응)
        raw_place_name_query = state.get("place_name_query")
        place_name_query: Optional[str] = (
            raw_place_name_query.strip()
            if isinstance(raw_place_name_query, str) and raw_place_name_query.strip()
            else None
        )
        place: Optional[Dict[str, Any]] = None
        place_search_method: Optional[str] = None
        if place_name_query:
            place = _match_from_nearby(place_name_query, nearby_places)
            if place:
                place_search_method = "nearby_match"
            elif site_id:
                place = _search_place(site_id, place_name_query)
                if place:
                    place_search_method = "pg_trgm"

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
            desc = place.get("description") or ""
            desc_part = f"\n위치설명 및 부가 설명: {desc}" if desc else ""
            dist = place.get("distanceM")
            dist_str = f"\n거리: 약 {dist:.0f}m" if isinstance(dist, (int, float)) else ""

            if user_language == "ko":
                system_msg = (
                    f"{persona_block}\n"
                    "규칙:\n"
                    "  - 아래 장소 정보를 바탕으로 1~2문장으로 위치를 안내하세요\n"
                    "  - '위치설명 및 부가 설명'이 있으면 반드시 포함해서 안내하세요\n"
                    "  - 목록에 없는 정보를 지어내지 마세요\n"
                    "  - 이모지나 특수문자는 사용하지 마세요\n"
                    "  - 마스코트의 슬로건이나 개성 문구가 있다면 answer가 아닌 sign_off에 넣으세요\n\n"
                    f"장소: {place_name} (카테고리: {place.get('category')}){dist_str}{desc_part}\n\n"
                    "반드시 아래 JSON 형식으로만 응답하세요 (마크다운 없이):\n"
                    '{"answer": "위치 안내 문구만", "sign_off": "슬로건/개성 문구 (없으면 빈 문자열)", "emotion": "GUIDING"}'
                )
            else:
                system_msg = (
                    f"{persona_block}\n"
                    "Rules:\n"
                    f"  - CRITICAL: Your entire answer MUST be in {lang_name}.\n"
                    f"  - Respond in {lang_name}, 1-2 sentences\n"
                    "  - If '위치설명' is provided, include that location detail in your answer\n"
                    "  - Do not invent location details not in the provided info\n"
                    "  - No emoji, no special characters\n"
                    "  - If the persona has a slogan or catchphrase, put it in sign_off, not in answer\n\n"
                    f"Place: {place_name} (category: {place.get('category')}){dist_str}{desc_part}\n\n"
                    "Return ONLY valid JSON (no markdown):\n"
                    '{"answer": "...", "sign_off": "slogan/catchphrase or empty string", "emotion": "GUIDING"}'
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
                sign_off = parsed.get("sign_off", "") if isinstance(parsed.get("sign_off"), str) else ""
                sign_off = sign_off.strip()
                raw_emotion = parsed.get("emotion", "GUIDING")
                emotion = raw_emotion if raw_emotion in _ALLOWED_EMOTIONS else "GUIDING"
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                extracted = _extract_json_from_mixed(raw)
                if extracted and isinstance(extracted.get("answer"), str) and extracted["answer"] and extracted["answer"] not in _ANSWER_PLACEHOLDERS:
                    answer_text = extracted["answer"]
                    sign_off = (extracted.get("sign_off") or "").strip()
                    raw_emotion = extracted.get("emotion", "GUIDING")
                    emotion = raw_emotion if raw_emotion in _ALLOWED_EMOTIONS else "GUIDING"
                    method = "llm_json_extracted"
                else:
                    answer_text = _text_before_json(raw)
                    sign_off = ""
                    method = "llm_raw_fallback"
                error_msg = str(e)
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
                sign_off = ""
                method = "llm_api_error"
                error_msg = str(e)

            if answer_text:
                if sign_off and answer_text.endswith(sign_off):
                    answer_text = answer_text[:-len(sign_off)].rstrip()
                if map_url:
                    guide = _MAP_GUIDE.get(user_language, _MAP_GUIDE["en"])
                    answer_text = f"{answer_text} {guide}"
                if sign_off:
                    answer_text = f"{answer_text} {sign_off}"
                trace["navigation"] = {
                    "status": "ok",
                    "method": method,
                    "search_method": place_search_method,
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
                "  - 이모지나 특수문자는 사용하지 마세요\n"
                "  - 마스코트의 슬로건이나 개성 문구가 있다면 answer가 아닌 sign_off에 넣으세요\n\n"
                f"주변 장소 목록:\n{places_text}\n\n"
                "반드시 아래 JSON 형식으로만 응답하세요 (마크다운 없이):\n"
                '{"answer": "위치 안내 문구만", "sign_off": "슬로건/개성 문구 (없으면 빈 문자열)", '
                '"place_id": <placeId 또는 null>, "emotion": "GUIDING", "category": "DIRECTION"}'
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
                "  - No emoji, no special characters\n"
                "  - If the persona has a slogan or catchphrase, put it in sign_off, not in answer\n\n"
                f"Nearby places:\n{places_text}\n\n"
                "Return ONLY valid JSON (no markdown):\n"
                '{"answer": "...", "sign_off": "slogan/catchphrase or empty string", '
                '"place_id": <placeId or null>, "emotion": "GUIDING", "category": "DIRECTION"}'
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
            sign_off = parsed.get("sign_off", "") if isinstance(parsed.get("sign_off"), str) else ""
            sign_off = sign_off.strip()
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
            extracted = _extract_json_from_mixed(raw)
            if extracted and isinstance(extracted.get("answer"), str) and extracted["answer"] and extracted["answer"] not in _ANSWER_PLACEHOLDERS:
                answer_text = extracted["answer"]
                sign_off = (extracted.get("sign_off") or "").strip()
                raw_place_id_ex = extracted.get("place_id")
                if isinstance(raw_place_id_ex, bool) or raw_place_id_ex is None:
                    place_id = None
                elif isinstance(raw_place_id_ex, int):
                    place_id = raw_place_id_ex
                elif isinstance(raw_place_id_ex, float) and raw_place_id_ex.is_integer():
                    place_id = int(raw_place_id_ex)
                raw_emotion = extracted.get("emotion", "GUIDING")
                emotion = raw_emotion if raw_emotion in _ALLOWED_EMOTIONS else "GUIDING"
                method = "llm_json_extracted"
            else:
                answer_text = _text_before_json(raw)
                sign_off = ""
                method = "llm_raw_fallback"
            error_msg = str(e)
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            sign_off = ""
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

        if sign_off and answer_text.endswith(sign_off):
            answer_text = answer_text[:-len(sign_off)].rstrip()
        if map_url:
            guide = _MAP_GUIDE.get(user_language, _MAP_GUIDE["en"])
            answer_text = f"{answer_text} {guide}"
        if sign_off:
            answer_text = f"{answer_text} {sign_off}"

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
