from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese",
    "ja": "Japanese", "fr": "French", "es": "Spanish",
}


def make_direct_llm_node(llm: OpenAILLM):
    """문서 검색 없이 LLM 이 직접 답변하는 노드 팩토리.

    추천·팁·코스·일반 상식처럼 RAG 문서가 필요 없는 질문에 사용.
    """

    def direct_llm_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language: str = state.get("user_language", "ko")
        site_id: int = state.get("site_id", 1)
        lang_name = _LANG_NAMES.get(user_language, user_language.upper())

        if user_language == "ko":
            system_prompt = (
                f"당신은 site_id={site_id} 관광지의 친절한 안내 도우미입니다.\n"
                "규칙:\n"
                "  - 한국어로 2~5문장, 음성으로 읽기 좋게 자연스럽게 답하세요\n"
                "  - 일반적인 관광 팁·추천·코스 등을 안내해 주세요\n"
                "  - 모르는 내용은 솔직히 모른다고 하세요"
            )
        else:
            system_prompt = (
                f"You are a helpful tourism guide assistant for site {site_id}.\n"
                f"Rules:\n"
                f"  - Answer in {lang_name}, 2-5 sentences, natural for speech\n"
                f"  - Provide general tourism tips, recommendations, or course suggestions\n"
                f"  - If you don't know, say so honestly"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            answer = llm.chat(messages, max_tokens=200)
        except Exception:
            answer = (
                "죄송합니다. 답변을 생성하지 못했습니다."
                if user_language == "ko"
                else "Sorry, I couldn't generate an answer."
            )

        trace = dict(state.get("trace") or {})
        trace["direct_llm"] = {"user_language": user_language}

        return {"answer_text": answer, "trace": trace}

    return direct_llm_node
