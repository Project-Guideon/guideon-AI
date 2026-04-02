from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState
'''아직 구현 안되어 있어요'''
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

        # ── 마스코트 페르소나 블록 동적 조립 ─────────────────────────────
        base_prompt = state.get("system_prompt") or ""
        name = state.get("mascot_name") or ""
        greeting = state.get("mascot_greeting") or ""
        style = (
            state.get("mascot_smalltalk_style")
            or state.get("mascot_base_persona")
            or ""
        )

        if user_language == "ko":
            persona_lines = []
            if base_prompt:
                persona_lines.append(base_prompt)
            if name:
                persona_lines.append(f"당신의 이름은 {name}입니다.")
            if greeting:
                persona_lines.append(f"인사말: {greeting}")
            if style:
                persona_lines.append(f"말투 지침: {style}")
            persona_block = "\n".join(persona_lines) if persona_lines else "당신은 관광지의 귀여운 마스코트 안내원입니다."

            system_prompt = (
                f"{persona_block}\n"
                "규칙:\n"
                "  - 2~3문장으로 짧게 답하세요\n"
                "  - 음성으로 읽기 좋게 자연스럽게 작성하세요\n"
                "  - 이모지나 특수문자는 사용하지 마세요"
            )
        else:
            persona_lines = []
            if base_prompt:
                persona_lines.append(
                    f"[Character setting (originally in Korean, for your reference only)]: {base_prompt}"
                )
            if name:
                persona_lines.append(f"Your name is {name}.")
            if greeting:
                persona_lines.append(f"Greeting: {greeting}")
            if style:
                persona_lines.append(
                    f"[Speech style instruction — written in Korean]: {style}\n"
                    f"→ Translate the above Korean style instruction into {lang_name} first, "
                    f"then follow it exactly in {lang_name}. "
                    f"If it says to add a word/phrase at the end of sentences, "
                    f"translate that word/phrase into {lang_name} and add it.\n"
                    f"- CRITICAL: The style MUST be visible in your response.\n"
                    f"- CRITICAL: Do NOT use the original Korean words — always translate them into {lang_name}."
                )
            persona_block = "\n".join(persona_lines) if persona_lines else f"You are a cute mascot guide at a tourist site."

            system_prompt = (
                f"{persona_block}\n"
                f"Rules:\n"
                f"  - CRITICAL: Your entire answer MUST be in {lang_name}. Do NOT include any Korean words or particles.\n"
                f"  - Respond in {lang_name}, 2-3 sentences\n"
                f"  - Keep it speech-friendly (no emoji, no special characters)"
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
        flow = list(trace.get("_flow") or [])
        flow.append("smalltalk")
        trace["_flow"] = flow
        trace["smalltalk"] = {"user_language": user_language, "check_result": check_result}

        return {"answer_text": answer, "check_result": check_result, "trace": trace}

    return smalltalk_node
