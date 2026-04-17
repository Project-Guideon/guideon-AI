from __future__ import annotations

from typing import List

from app.graph.state import GraphState


ALLOWED_INTENTS = {"rag", "smalltalk", "event", "struct_db"}


def intent_router(state: GraphState) -> str:
    """intent_gate 결과(ranking)로 첫 번째 의도의 다음 노드를 결정.

    RAG 일 때 language_code 분기도 여기서 처리.

    Returns:
        "smalltalk"    → smalltalk 노드
        "event"        → event 노드
        "struct_db"    → struct_db 노드
        "retrieve"     → RAG (KO) 경로
        "translate_ko" → RAG (Foreign) 경로
    """
    ranking: List[str] = state.get("intent_ranking") or ["rag"]
    index: int = state.get("current_intent_index", 0)
    language_code: str = state.get("language_code", "ko")

    if not isinstance(index, int) or index < 0 or index >= len(ranking):
        index = 0

    intent = ranking[index]

    if intent not in ALLOWED_INTENTS:
        intent = "rag"

    if intent == "rag":
        # translate_ko 단계 제거: intent_gate에서 retrieval_query_ko를 이미 생성함
        return "retrieve"

    return intent  # "smalltalk" | "event" | "struct_db"
