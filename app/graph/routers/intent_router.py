from __future__ import annotations

from app.graph.state import GraphState


def intent_router(state: GraphState) -> str:
    """intent_gate 결과로 다음 노드를 결정.

    Returns:
        "smalltalk"    → smalltalk 노드
        "info_request" → infotype_gate 노드
    """
    return state.get("intent", "info_request")
