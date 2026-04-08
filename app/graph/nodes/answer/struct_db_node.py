from __future__ import annotations

import json
from typing import List, Dict, Any

from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState
from app.graph.nodes.utils import LANG_NAMES, build_messages, append_trace_flow, build_persona_block


def make_struct_db_node(llm: OpenAILLM):
    """장소/위치 조회 노드 팩토리.

    Spring Boot Core 가 PostGIS 로 조립한 nearby_places context 를 활용해
    GPT 가 질문에 맞는 장소를 선택하고 자연어 답변을 생성한다.

    - sameZone=True 인 장소 우선 고려 (Spring Boot Core 가 이미 정렬해서 전달)
    - 장소가 없으면 check_result="bad" 로 RAG fallback
    """

    def struct_db_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language: str = state.get("user_language", "ko")
        lang_name = LANG_NAMES.get(user_language, user_language.upper())
        nearby_places: List[Dict[str, Any]] = state.get("nearby_places") or []
        system_prompt: str = state.get("system_prompt") or ""

        trace = append_trace_flow(state, "struct_db")

        # nearby_places 없으면 RAG fallback
        if not nearby_places:
            trace["struct_db"] = {"status": "no_context", "query": text}
            return {"answer_text": "", "check_result": "bad", "trace": trace}

        # 장소 목록 포맷 (GPT 프롬프트용)
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

        # 마스코트 페르소나 블록 조립 (ko/foreign 분기는 build_persona_block 내부에서 처리)
        name = state.get("mascot_name") or ""
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
            name=name,
            ko_fallback="당신은 관광지의 친절한 위치 안내원입니다.",
            foreign_fallback="You are a friendly tourist guide assistant.",
        )

        # 노드별 고유 규칙 + 장소 목록 + JSON 형식 지정
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
                f"  - CRITICAL: Your entire answer MUST be in {lang_name}. Do NOT include any Korean words or particles.\n"
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

        # system → 이전 대화 내역(chat_history) → 현재 질문 순으로 조립
        messages = build_messages(state, system_msg, text)

        method = "llm"
        error_msg = ""
        answer_text = ""
        place_id = None
        emotion = "GUIDING"
        category = "DIRECTION"
        raw = ""

        _allowed_emotions = {"GUIDING", "HAPPY", "THINKING", "SORRY", "EXCITED"}
        _allowed_categories = {"DIRECTION", "HOURS", "FACILITY", "HISTORY", "GENERAL", "ERROR"}

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
            else:
                place_id = None
            raw_emotion = parsed.get("emotion", "GUIDING")
            emotion = raw_emotion if raw_emotion in _allowed_emotions else "GUIDING"
            raw_category = parsed.get("category", "DIRECTION")
            category = raw_category if raw_category in _allowed_categories else "DIRECTION"
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # JSON 파싱 실패 → raw 자체를 답변으로 사용
            answer_text = raw
            method = "llm_raw_fallback"
            error_msg = str(e)
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            method = "llm_api_error"
            error_msg = str(e)

        if not answer_text:
            trace["struct_db"] = {"status": "llm_failed", "method": method, "error": error_msg}
            return {"answer_text": "", "check_result": "bad", "trace": trace}

        trace["struct_db"] = {
            "status": "ok",
            "method": method,
            "place_id": place_id,
            "places_count": len(nearby_places),
        }
        if error_msg:
            trace["struct_db"]["error"] = error_msg

        return {
            "answer_text": answer_text,
            "place_id": place_id,
            "emotion": emotion,
            "category": category,
            "check_result": "good",
            "trace": trace,
        }

    return struct_db_node
