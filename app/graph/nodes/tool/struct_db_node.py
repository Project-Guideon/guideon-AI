from __future__ import annotations

import json
from typing import List, Dict, Any

from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState


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
        nearby_places: List[Dict[str, Any]] = state.get("nearby_places") or []
        system_prompt: str = state.get("system_prompt") or ""

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("struct_db")
        trace["_flow"] = flow

        # nearby_places 없으면 RAG fallback
        if not nearby_places:
            trace["struct_db"] = {"status": "no_context", "query": text}
            return {"answer_text": "", "check_result": "bad", "trace": trace}

        # 장소 목록 포맷 (GPT 프롬프트용)
        places_lines = []
        for p in nearby_places:
            same = "✓ 같은 구역" if p.get("sameZone") else "다른 구역"
            dist = p.get("distanceM")
            dist_str = f"{dist:.0f}m" if dist is not None else "거리 미상"
            desc = p.get("description") or ""
            places_lines.append(
                f"- placeId={p.get('placeId')} [{p.get('name')}] "
                f"카테고리: {p.get('category')} | {same} | {dist_str} | {desc}"
            )
        places_text = "\n".join(places_lines)

        lang_instruction = {
            "ko": "한국어로 답변하세요.",
            "en": "Answer in English.",
            "zh": "请用中文回答。",
            "ja": "日本語で答えてください。",
        }.get(user_language, "Answer in the same language as the question.")

        character_instruction = f"\n{system_prompt}" if system_prompt else ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly guide at a tourist attraction."
                    f"{character_instruction}\n"
                    "Based on the nearby places list, answer the user's location/direction question.\n"
                    "Prioritize places marked '✓ 같은 구역' (same zone) and shorter distances.\n"
                    f"{lang_instruction}\n"
                    "Be concise and friendly. Mention the place name and distance.\n\n"
                    f"Nearby places:\n{places_text}\n\n"
                    "Respond ONLY with valid JSON (no markdown, no extra text):\n"
                    "{\n"
                    '  "answer": "자연어 답변 (1~2문장)",\n'
                    '  "place_id": <가장 관련 있는 placeId (int) 또는 null>,\n'
                    '  "emotion": "GUIDING",\n'
                    '  "category": "DIRECTION"\n'
                    "}"
                ),
            },
            {"role": "user", "content": text},
        ]

        method = "llm"
        error_msg = ""
        answer_text = ""
        place_id = None
        emotion = "GUIDING"
        category = "DIRECTION"
        raw = ""

        try:
            raw = llm.chat(messages, max_tokens=200)
            parsed = json.loads(raw)
            answer_text = parsed.get("answer", "")
            place_id = parsed.get("place_id")
            emotion = parsed.get("emotion", "GUIDING")
            category = parsed.get("category", "DIRECTION")
        except json.JSONDecodeError as e:
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
