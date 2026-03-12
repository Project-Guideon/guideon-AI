from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese",
    "ja": "Japanese", "fr": "French", "es": "Spanish",
}


def make_smalltalk_node(llm: OpenAILLM):
    """일상 대화(인사·감정·잡담) 응답 노드 팩토리.

    - user_language 에 따라 자동 언어 분기
    - 마스코트 페르소나, 2~3문장, TTS 친화적
    """

    def smalltalk_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language: str = state.get("user_language", "ko")
        lang_name = _LANG_NAMES.get(user_language, user_language.upper())

        if user_language == "ko":
            system_prompt = (
                "당신은 관광지의 귀여운 마스코트 안내원입니다.\n"
                "규칙:\n"
                "  - 친근하고 따뜻하게, 2~3문장으로 짧게 답하세요\n"
                "  - 음성으로 읽히는 것을 고려해 자연스럽게 작성하세요\n"
                "  - 이모지나 특수문자는 사용하지 마세요\n"
                "  - 과하게 흥분하거나 과장하지 말고 자연스럽게 대화하세요"
            )
        else:
            system_prompt = (
                f"You are a friendly mascot guide at a tourism site.\n"
                f"Rules:\n"
                f"  - Respond in {lang_name}, warmly and naturally, 2-3 sentences\n"
                f"  - Keep it speech-friendly (no emoji, no special characters)\n"
                f"  - Be natural and friendly, not overly excited"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        check_result = "good"
        try:
            answer = llm.chat(messages, max_tokens=100)
        except Exception:
            answer = ""
            check_result = "bad"

        trace = dict(state.get("trace") or {})
        trace["smalltalk"] = {"user_language": user_language, "check_result": check_result}

        return {"answer_text": answer, "check_result": check_result, "trace": trace}

    return smalltalk_node
