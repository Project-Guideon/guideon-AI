from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState


_LANG_NAMES = {
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
}


def _serialize_daily_info(idx: int, d: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": d.get("id", idx + 1),
        "place_name": d.get("placeName", ""),
        "info_type": d.get("infoType", ""),
        "content": d.get("content", ""),
        "start_date": d.get("startDate", ""),
        "end_date": d.get("endDate", ""),
        "start_time": d.get("startTime", ""),
        "end_time": d.get("endTime", ""),
        "all_day": d.get("allDay", False),
    }


def _build_event_messages(
    query: str,
    user_language: str,
    now_str: str,
    candidate_infos: list[dict[str, Any]],
    system_prompt_base: str,
    answer_style: str,
) -> list[dict[str, str]]:
    candidate_json = json.dumps(candidate_infos, ensure_ascii=False, indent=2)

    character_lines = []
    if system_prompt_base:
        character_lines.append(system_prompt_base)
    if answer_style:
        character_lines.append(f"답변 스타일: {answer_style}")
    character_instruction = "\n".join(character_lines).strip()

    lang_name = _LANG_NAMES.get(user_language, user_language.upper())
    lang_instruction = (
        "answer는 반드시 한국어로 작성하세요. 2~3문장으로 자연스럽고 음성 안내처럼 답변하세요. "
        "이모지나 특수문자는 사용하지 마세요."
        if user_language == "ko"
        else f'The "answer" field must be written in {lang_name}. '
        f"Use 2-3 natural, speech-friendly sentences. No emoji, no special characters."
    )

    return [
        {
            "role": "system",
            "content": (
                "You are the event/operation information answering module for a multilingual tourism chatbot.\n"
                "Use ONLY the provided candidate records.\n"
                "Do not invent facts.\n"
                "Select only records relevant to the user's question.\n"
                "If nothing is relevant, return no_match=true.\n"
                "If date/time fields exist, reflect them clearly in the answer.\n"
                "If the question is about restrictions, closures, repairs, or route changes, "
                "explain the affected place and restriction clearly.\n"
                f"Current datetime: {now_str}\n\n"
                f"{character_instruction}\n\n"
                f"{lang_instruction}\n\n"
                "Return ONLY valid JSON in this exact format:\n"
                '{'
                '"no_match": false, '
                '"selected_ids": [1, 2], '
                '"answer": "..."'
                '}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"user_language: {user_language}\n"
                f"user_question: {query}\n\n"
                f"candidate_records:\n{candidate_json}"
            ),
        },
    ]


def make_event_node(llm: OpenAILLM):
    def event_node(state: GraphState) -> dict:
        text: str = (state.get("normalized_text") or "").strip()
        user_language: str = state.get("user_language", "ko")
        daily_infos: list[dict[str, Any]] = state.get("daily_infos") or []
        site_id: int = state.get("site_id", 1)

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("event")
        trace["_flow"] = flow

        if not daily_infos:
            trace["event"] = {
                "status": "no_context",
                "query": text,
                "site_id": site_id,
                "infos_count": 0,
            }
            return {
                "answer_text": "",
                "check_result": "bad",
                "trace": trace,
            }

        candidate_infos = [
            _serialize_daily_info(idx, d)
            for idx, d in enumerate(daily_infos[:8])
        ]

        system_prompt_base: str = state.get("system_prompt") or ""
        answer_style: str = (
            state.get("mascot_event_style")
            or state.get("mascot_base_persona")
            or ""
        )

        messages = _build_event_messages(
            query=text,
            user_language=user_language,
            now_str=datetime.now().strftime("%Y-%m-%d %H:%M"),
            candidate_infos=candidate_infos,
            system_prompt_base=system_prompt_base,
            answer_style=answer_style,
        )

        raw = ""
        answer_text = ""
        selected_ids: list[int] = []
        method = "llm"
        error_msg = ""
        check_result = "bad"

        try:
            raw = llm.chat(messages, max_tokens=250)
            parsed = json.loads(raw)

            no_match = bool(parsed.get("no_match", True))
            answer_text = (parsed.get("answer") or "").strip()
            selected_ids_raw = parsed.get("selected_ids", [])

            if isinstance(selected_ids_raw, list):
                selected_ids = [
                    int(x) for x in selected_ids_raw
                    if isinstance(x, int) or str(x).isdigit()
                ]

            if (not no_match) and answer_text:
                check_result = "good"
            else:
                check_result = "bad"

        except json.JSONDecodeError as e:
            method = "llm_invalid_json"
            error_msg = str(e)

        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            method = "llm_api_error"
            error_msg = str(e)

        except Exception as e:
            method = "llm_error"
            error_msg = str(e)

        trace["event"] = {
            "status": "ok" if check_result == "good" else "no_match",
            "query": text,
            "site_id": site_id,
            "infos_count": len(daily_infos),
            "candidate_count": len(candidate_infos),
            "selected_ids": selected_ids,
            "method": method,
            "raw": raw,
        }
        if error_msg:
            trace["event"]["error"] = error_msg

        return {
            "answer_text": answer_text,
            "check_result": check_result,
            "trace": trace,
        }

    return event_node