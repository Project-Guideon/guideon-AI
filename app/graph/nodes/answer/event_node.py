from __future__ import annotations

from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese",
    "ja": "Japanese", "fr": "French", "es": "Spanish",
}


def make_event_node(llm: OpenAILLM):
    """이벤트/운영정보 응답 노드 팩토리.

    - Spring Boot Core 가 조립한 daily_infos(dailyInfos context) 를 활용
    - daily_infos 없으면 check_result="bad" → RAG fallback
    - system_prompt + mascot_answer_style 로 마스코트 캐릭터 유지
    """

    def event_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language: str = state.get("user_language", "ko")
        daily_infos: list = state.get("daily_infos") or []
        site_id: int = state.get("site_id", 1)

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("event")
        trace["_flow"] = flow

        # daily_infos 없으면 RAG fallback
        if not daily_infos:
            trace["event"] = {"status": "no_context", "query": text, "site_id": site_id}
            return {"answer_text": "", "check_result": "bad", "trace": trace}

        # 운영정보 포맷
        info_lines = [
            f"- [{d.get('placeName', '')}] {d.get('infoType', '')}: {d.get('content', '')}"
            for d in daily_infos
        ]
        context_str = "\n".join(info_lines)

        # 마스코트 캐릭터 적용
        system_prompt_base: str = state.get("system_prompt") or ""
        answer_style: str = (
            state.get("mascot_event_style")
            or state.get("mascot_base_persona")
            or ""
        )
        character_lines = []
        if system_prompt_base:
            character_lines.append(system_prompt_base)
        if answer_style:
            character_lines.append(f"답변 스타일: {answer_style}")
        character_instruction = "\n" + "\n".join(character_lines) if character_lines else ""

        lang_name = _LANG_NAMES.get(user_language, user_language.upper())
        lang_instruction = (
            "한국어로 2~3문장, 음성으로 읽기 좋게 자연스럽게 작성하세요. 이모지나 특수문자는 사용하지 마세요."
            if user_language == "ko"
            else f"Respond in {lang_name}, 2-3 sentences, speech-friendly (no emoji, no special characters)."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a friendly guide at a tourist attraction.{character_instruction}\n\n"
                    f"운영 정보:\n{context_str}\n\n"
                    f"{lang_instruction}"
                ),
            },
            {"role": "user", "content": text},
        ]

        answer_text = ""
        method = "llm"
        error_msg = ""

        try:
            answer_text = llm.chat(messages, max_tokens=150)
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            method = "llm_api_error"
            error_msg = str(e)
        except Exception as e:
            method = "llm_error"
            error_msg = str(e)

        if not answer_text:
            trace["event"] = {"status": "llm_failed", "method": method, "error": error_msg}
            return {"answer_text": "", "check_result": "bad", "trace": trace}

        trace["event"] = {
            "status": "ok",
            "method": method,
            "infos_count": len(daily_infos),
        }
        if error_msg:
            trace["event"]["error"] = error_msg

        return {"answer_text": answer_text, "check_result": "good", "trace": trace}

    return event_node
