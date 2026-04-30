from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState
from app.graph.nodes.utils import LANG_NAMES, build_messages, append_trace_flow, build_persona_block, get_language

_NO_RESULT_MSG = {
    "ko": "관련 정보를 찾을 수 없습니다. 더 구체적으로 말씀해 주시겠어요?",
    "en": "I couldn't find any relevant information. Could you be more specific?",
    "zh": "未能找到相关信息，请您再具体描述一下好吗？",
    "ja": "関連する情報が見つかりませんでした。もう少し具体的に教えていただけますか？",
    "fr": "Je n'ai trouvé aucune information pertinente. Pourriez-vous préciser votre question ?",
    "es": "No encontré información relevante. ¿Podría ser más específico?",
}

_ERROR_MSG = {
    "ko": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
    "en": "Sorry, an error occurred while generating the answer.",
    "zh": "抱歉，生成回答时出现了错误。",
    "ja": "申し訳ありません。回答の生成中にエラーが発生しました。",
    "fr": "Désolé, une erreur est survenue lors de la génération de la réponse.",
    "es": "Lo siento, ocurrió un error al generar la respuesta.",
}


def make_answer_generate_node(llm: OpenAILLM):
    """답변 생성 노드 팩토리.

    - KO  : 한국어 context → 한국어 답변
    - Foreign : 한국어 context → user_language 로 직접 생성 (번역 단계 없음)
    """

    def answer_generate_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        chunks: list = state.get("retrieved_chunks") or []
        user_language = get_language(state)
        lang_name = LANG_NAMES.get(user_language, user_language.upper())

        # 빈 검색 결과 → 고정 응답으로 즉시 종료
        if not chunks:
            answer = _NO_RESULT_MSG.get(user_language, _NO_RESULT_MSG["en"])
            trace = append_trace_flow(state, "answer_generate")
            trace["answer_generate"] = {
                "user_language": user_language,
                "chunks_used": 0,
                "answer_length": len(answer),
            }
            return {"answer_text": answer, "category": "INFORMATION", "trace": trace}

        # ── 컨텍스트 조립 ──────────────────────────────────────────────
        def _format_doc(i: int, c: dict) -> str:
            title = c.get("section_title", "")
            header = f"[문서 {i + 1}] {title}".strip() if title else f"[문서 {i + 1}]"
            return f"{header}\n{c['content']}"

        context_str = "\n\n".join(
            _format_doc(i, c) for i, c in enumerate(chunks)
        )

        # 마스코트 페르소나 블록 조립 (ko/foreign 분기는 build_persona_block 내부에서 처리)
        system_prompt_base: str = state.get("system_prompt") or ""
        RAG_style: str = (
            state.get("mascot_RAG_style")
            or state.get("mascot_base_persona")
            or ""
        )
        persona_block = build_persona_block(
            base_prompt=system_prompt_base,
            style=RAG_style,
            user_language=user_language,
            lang_name=lang_name,
            ko_style_label="답변 스타일",
            ko_fallback="당신은 관광 안내 음성 도우미입니다.",
            foreign_fallback="You are a tourism guide voice assistant.",
        )

        # 노드별 고유 규칙 + context 조립
        if user_language == "ko":
            system_prompt = (
                f"{persona_block}\n"
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
            system_prompt = (
                f"{persona_block}\n"
                f"Answer in {lang_name} using the Korean reference documents provided.\n"
                "Rules:\n"
                f"  - CRITICAL: Your entire answer MUST be in {lang_name}. Do NOT include any Korean words or particles.\n"
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

        # system → 이전 대화 내역(chat_history) → 현재 질문 순으로 조립
        messages = build_messages(state, system_prompt, user_msg)

        try:
            answer = llm.chat(messages, max_tokens=150)
        except Exception:
            answer = _ERROR_MSG.get(user_language, _ERROR_MSG["en"])

        trace = append_trace_flow(state, "answer_generate")
        trace["answer_generate"] = {
            "user_language": user_language,
            "chunks_used": len(chunks),
            "answer_length": len(answer),
        }

        return {"answer_text": answer, "category": "INFORMATION", "trace": trace}

    return answer_generate_node
