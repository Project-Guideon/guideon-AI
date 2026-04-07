from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese",
    "ja": "Japanese", "fr": "French", "es": "Spanish",
}


def make_clarify_node(llm: OpenAILLM):
    """Clarifying Question 생성 노드 팩토리.

    answer_check 에서 bad 최종 판정 시 호출.
    더 구체적인 정보를 요청하는 짧은 질문을 user_language 로 생성.
    """

    def clarify_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language: str = state.get("user_language", "ko")
        lang_name = _LANG_NAMES.get(user_language, user_language.upper())

        if user_language == "ko":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "당신은 관광 안내 음성 도우미입니다.\n"
                        "사용자의 질문에 대한 정보를 찾지 못했습니다.\n"
                        "더 정확한 답변을 위해 추가 정보를 요청하는 짧고 자연스러운 질문을 1~2문장으로 만드세요.\n"
                        "예: '어느 건물/구역을 기준으로 찾으시나요?' / '구체적인 장소명을 알 수 있을까요?'"
                    ),
                },
                {"role": "user", "content": f"사용자 원래 질문: {text}"},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a tourism guide voice assistant.\n"
                        f"You could not find relevant information for the user's question.\n"
                        f"Generate a SHORT clarifying question in {lang_name} (1-2 sentences) "
                        f"to ask the user for more specific details.\n"
                        f"Example: 'Could you tell me which building or area you are referring to?'"
                    ),
                },
                {"role": "user", "content": f"User's original question: {text}"},
            ]

        try:
            clarify_text = llm.chat(messages, max_tokens=80)
        except Exception:
            clarify_text = (
                "죄송합니다. 좀 더 구체적으로 말씀해 주시겠어요?"
                if user_language == "ko"
                else "Sorry, could you please provide more details?"
            )

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("clarify")
        trace["_flow"] = flow
        trace["clarify"] = {"user_language": user_language, "clarify_text": clarify_text}

        return {"answer_text": clarify_text, "trace": trace}

    return clarify_node
