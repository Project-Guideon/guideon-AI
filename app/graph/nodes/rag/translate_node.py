from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState


def make_translate_node(llm: OpenAILLM):
    """외국어 → 한국어 검색 쿼리 변환 노드 팩토리.

    핵심 정책:
    - retrieval_query_ko(검색용 한국어 쿼리)만 생성
    - user_language / normalized_text 는 절대 변경하지 않음
    - 답변은 나중에 answer_generate 에서 user_language 로 직접 생성
    """

    def translate_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language: str = state.get("user_language", "ko")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a translation assistant for a Korean tourism chatbot.\n"
                    "Translate the user's question into a SHORT Korean search query (keywords only).\n"
                    "Rules:\n"
                    "  - Output ONLY the Korean search keywords, no explanation\n"
                    "  - Keep it concise (5-15 Korean words max)\n"
                    "  - Preserve proper nouns (place names, monument names) if mentioned\n"
                    "  - Do NOT answer the question, only translate for search purposes"
                ),
            },
            {
                "role": "user",
                "content": f"Translate this {user_language} question into Korean search keywords:\n{text}",
            },
        ]

        try:
            retrieval_query_ko = llm.chat(messages, max_tokens=60).strip()
        except Exception:
            # 번역 실패 시 원문 그대로 사용 (한국어 RAG 가 어느 정도 처리 가능)
            retrieval_query_ko = text

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("translate_ko")
        trace["_flow"] = flow
        trace["translate"] = {
            "original": text,
            "user_language": user_language,
            "retrieval_query_ko": retrieval_query_ko,
        }

        return {"retrieval_query_ko": retrieval_query_ko, "trace": trace}

    return translate_node
