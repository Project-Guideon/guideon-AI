from __future__ import annotations

from typing import List

from app.graph.state import GraphState


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

    intent = ranking[index] if index < len(ranking) else "rag"

    if intent == "rag":
        return "retrieve" if language_code == "ko" else "translate_ko"

    return intent  # "smalltalk" | "event" | "struct_db"
