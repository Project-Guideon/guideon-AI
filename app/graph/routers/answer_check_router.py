from __future__ import annotations

from app.graph.state import GraphState


def answer_check_router(state: GraphState) -> str:
    """answer_check 결과로 다음 노드를 결정.

    Returns:
        "fallback_dispatch" → 성공(good) 또는 최종 실패(bad) 모두 fallback_dispatch 로 수렴
        "retry"             → top_k 증가 + MMR 재검색 루프백
    """
    result = state.get("check_result", "bad")
    if result == "retry":
        return "retry"
    # good / bad 모두 fallback_dispatch 로 보냄
    # fallback_dispatch 가 good 이면 종료, bad 이면 다음 의도로 이동
    return "fallback_dispatch"
