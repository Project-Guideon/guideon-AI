from __future__ import annotations

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


def make_answer_generate_node(llm: OpenAILLM):
    """답변 생성 노드 팩토리.

    - KO  : 한국어 context → 한국어 답변
    - Foreign : 한국어 context → user_language 로 직접 생성 (번역 단계 없음)
    """

    def answer_generate_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        chunks: list = state.get("retrieved_chunks") or []
        user_language: str = state.get("user_language", "ko")

        # ── 컨텍스트 조립 ──────────────────────────────────────────────
        if chunks:
            context_str = "\n\n".join(
                f"[문서 {i + 1}]\n{c['content']}" for i, c in enumerate(chunks)
            )
        else:
            context_str = ""

        # ── 프롬프트 분기 (KO / Foreign) ───────────────────────────────
        if user_language == "ko":
            system_prompt = (
                "당신은 관광 안내 음성 도우미입니다.\n"
                "제공된 참고 정보(context)만 근거로 답변하세요.\n"
                "규칙:\n"
                "  - 한국어로 2~5문장, 음성으로 읽기 좋게 자연스럽게 작성\n"
                "  - context 에 없는 내용은 추측하지 말 것\n"
                "  - 정보가 없으면 '관련 정보를 찾을 수 없습니다. 더 구체적으로 말씀해 주시겠어요?'라고만 답할 것\n"
                "  - 출처 문서명은 읽어주지 않아도 됨"
            )
            user_msg = (
                f"질문: {text}\n\n"
                f"참고 정보:\n{context_str}\n\n"
                "위 정보를 바탕으로 답변해 주세요."
            )
        else:
            lang_name = _LANG_NAMES.get(user_language, user_language.upper())
            system_prompt = (
                f"You are a tourism guide voice assistant.\n"
                f"Answer in {lang_name} using the Korean reference documents provided.\n"
                "Rules:\n"
                f"  - Respond in {lang_name}, 2-5 sentences, natural for speech\n"
                "  - Base your answer ONLY on the Korean context below\n"
                "  - Do NOT translate the answer from Korean — generate directly in the target language\n"
                "  - If the context does not contain relevant information, say so politely in "
                f"{lang_name} and ask for clarification\n"
                "  - Do not mention source document names"
            )
            user_msg = (
                f"Question ({lang_name}): {text}\n\n"
                f"Korean reference documents:\n{context_str}\n\n"
                f"Answer in {lang_name}:"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        try:
            answer = llm.chat(messages, max_tokens=150)
        except Exception as e:
            answer = (
                "죄송합니다. 답변 생성 중 오류가 발생했습니다."
                if user_language == "ko"
                else "Sorry, an error occurred while generating the answer."
            )

        trace = dict(state.get("trace") or {})
        trace["answer_generate"] = {
            "user_language": user_language,
            "chunks_used": len(chunks),
            "answer_length": len(answer),
        }

        return {"answer_text": answer, "trace": trace}

    return answer_generate_node
