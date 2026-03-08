from __future__ import annotations

from app.graph.state import GraphState


def answer_check_router(state: GraphState) -> str:
    """answer_check 결과로 다음 노드를 결정.

    Returns:
        "good"  → tts_builder (답변 품질 통과)
        "retry" → retrieve    (top_k 증가 + MMR 재검색 루프백)
        "bad"   → clarify     (최대 재시도 초과, clarifying question fallback)
    """
    return state.get("check_result", "bad")
