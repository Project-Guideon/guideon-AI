from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState
from app.graph.nodes.utils import LANG_NAMES, build_messages, append_trace_flow, build_persona_block


def make_smalltalk_node(llm: OpenAILLM):
    """일상 대화(인사·감정·잡담) 응답 노드 팩토리.

    - user_language 에 따라 자동 언어 분기
    - 마스코트 페르소나, 2~3문장, TTS 친화적
    """

    def smalltalk_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        user_language: str = state.get("user_language", "ko")
        lang_name = LANG_NAMES.get(user_language, user_language.upper())

        base_prompt = state.get("system_prompt") or ""
        name = state.get("mascot_name") or ""
        greeting = state.get("mascot_greeting") or ""
        style = (
            state.get("mascot_smalltalk_style")
            or state.get("mascot_base_persona")
            or ""
        )

        # 마스코트 페르소나 블록 조립 (ko/foreign 분기는 build_persona_block 내부에서 처리)
        persona_block = build_persona_block(
            base_prompt=base_prompt,
            style=style,
            user_language=user_language,
            lang_name=lang_name,
            name=name,
            greeting=greeting,
            ko_fallback="당신은 관광지의 귀여운 마스코트 안내원입니다.",
            foreign_fallback="You are a cute mascot guide at a tourist site.",
        )

        # 노드별 고유 규칙 추가
        if user_language == "ko":
            system_prompt = (
                f"{persona_block}\n"
                "규칙:\n"
                "  - 2~3문장으로 짧게 답하세요\n"
                "  - 음성으로 읽기 좋게 자연스럽게 작성하세요\n"
                "  - 이모지나 특수문자는 사용하지 마세요"
            )
        else:
            system_prompt = (
                f"{persona_block}\n"
                "Rules:\n"
                f"  - CRITICAL: Your entire answer MUST be in {lang_name}. Do NOT include any Korean words or particles.\n"
                f"  - Respond in {lang_name}, 2-3 sentences\n"
                "  - Keep it speech-friendly (no emoji, no special characters)"
            )

        # system → 이전 대화 내역(chat_history) → 현재 질문 순으로 조립
        messages = build_messages(state, system_prompt, text)

        check_result = "good"
        try:
            answer = llm.chat(messages, max_tokens=100)
        except Exception:
            answer = ""
            check_result = "bad"

        trace = append_trace_flow(state, "smalltalk")
        trace["smalltalk"] = {"user_language": user_language, "check_result": check_result}

        return {"answer_text": answer, "check_result": check_result, "trace": trace}

    return smalltalk_node
