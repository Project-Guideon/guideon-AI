from __future__ import annotations

from app.graph.state import GraphState

# 언어 코드 → 영문명 매핑 (전 노드 공통)
LANG_NAMES: dict[str, str] = {
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
}


def build_messages(state: GraphState, system_content: str, user_content: str) -> list[dict]:
    """LLM 메시지 리스트 조립.

    system → 이전 대화 내역(chat_history) → 현재 질문 순으로 조립.
    chat_history는 Redis에서 로드되어 state에 담겨 있음.
    load_chat_history에서 1차 검증을 하지만, 방어적으로 재검증.
    """
    raw_history = state.get("chat_history") or []
    history: list[dict[str, str]] = [
        {"role": item["role"], "content": item["content"]}
        for item in raw_history
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    return [
        {"role": "system", "content": system_content},
        *history,
        {"role": "user", "content": user_content},
    ]


def get_language(state: GraphState) -> str:
    """state에서 user_language를 안전하게 추출.

    None이거나 빈 문자열이면 "ko" 반환.
    """
    lang = state.get("user_language", "ko")
    return lang if isinstance(lang, str) and lang else "ko"


def append_trace_flow(state: GraphState, node_name: str) -> dict:
    """trace dict에 node_name을 _flow에 추가하여 반환.

    모든 노드에서 공통으로 쓰이는 trace 초기화 보일러플레이트를 대체.
    """
    trace = dict(state.get("trace") or {})
    flow = list(trace.get("_flow") or [])
    flow.append(node_name)
    trace["_flow"] = flow
    return trace


def build_foreign_style_line(style: str, lang_name: str) -> str:
    """한국어 말투 지침을 target language로 번역·적용하도록 안내하는 문자열 생성.

    smalltalk, struct_db, event, RAG 노드에서 외국어 응답 시 공통 사용.
    """
    return (
        f"[Speech style instruction — written in Korean]: {style}\n"
        f"→ Translate the above Korean style instruction into {lang_name} first, "
        f"then follow it exactly in {lang_name}. "
        f"If it says to add a word/phrase at the end of sentences, "
        f"translate that word/phrase into {lang_name} and add it.\n"
        f"- CRITICAL: The style MUST be visible in your response.\n"
        f"- CRITICAL: Do NOT use the original Korean words — always translate them into {lang_name}."
    )


def build_persona_block(
    base_prompt: str,
    style: str,
    user_language: str,
    lang_name: str,
    name: str = "",
    greeting: str = "",
    ko_style_label: str = "말투 지침",
    ko_fallback: str = "",
    foreign_fallback: str = "",
) -> str:
    """마스코트 페르소나 블록 조립.

    ko:      base_prompt → name → greeting → "{ko_style_label}: {style}"
    foreign: "[Character setting...]" → name → "[Speech style instruction...]"
    조립된 줄이 없으면 fallback 문자열 반환.

    Args:
        base_prompt:    마스코트 기본 캐릭터 설명 (system_prompt)
        style:          노드별 말투/답변 스타일 지침
        user_language:  사용자 언어 코드 ("ko" / 기타)
        lang_name:      언어 영문명 (예: "English")
        name:           마스코트 이름 (없으면 생략)
        greeting:       인사말 (없으면 생략)
        ko_style_label: 한국어 스타일 라벨 (기본 "말투 지침", RAG/event는 "답변 스타일")
        ko_fallback:    페르소나 정보 없을 때 한국어 기본값
        foreign_fallback: 페르소나 정보 없을 때 외국어 기본값
    """
    if user_language == "ko":
        lines = []
        if base_prompt:
            lines.append(base_prompt)
        if name:
            lines.append(f"당신의 이름은 {name}입니다.")
        if greeting:
            lines.append(f"인사말: {greeting}")
        if style:
            lines.append(f"{ko_style_label}: {style}")
        return "\n".join(lines) if lines else ko_fallback
    else:
        lines = []
        if base_prompt:
            lines.append(
                f"[Character setting (originally in Korean, for your reference only)]: {base_prompt}"
            )
        if name:
            lines.append(f"Your name is {name}.")
        if greeting:
            lines.append(f"Greeting: {greeting}")
        if style:
            lines.append(build_foreign_style_line(style, lang_name))
        return "\n".join(lines) if lines else foreign_fallback
