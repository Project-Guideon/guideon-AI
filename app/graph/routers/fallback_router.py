from __future__ import annotations

from typing import List

from app.graph.state import GraphState


# ── fallback_dispatch 노드 ────────────────────────────────────────────────────

def fallback_dispatch_node(state: GraphState) -> dict:
    """모든 분기의 결과가 수렴하는 노드.

    check_result 를 확인해서 다음 행선지를 결정한다.
    - "good" → fallback_next = "done" (성공, 종료)
    - "bad"  → current_intent_index 증가 → 다음 의도로 재라우팅
    - 순위 소진 → fallback_next = "clarify"
    """
    check_result: str = state.get("check_result", "bad")
    ranking: List[str] = state.get("intent_ranking") or []
    index: int = state.get("current_intent_index", 0)
    language_code: str = state.get("language_code", "ko")

    trace = dict(state.get("trace") or {})

    if check_result == "good":
        trace["fallback_dispatch"] = {"action": "done", "index": index}
        return {"fallback_next": "done", "trace": trace}

    # ── 다음 의도로 이동 ──────────────────────────────────────────────
    next_index = index + 1

    if next_index >= len(ranking):
        # 모든 의도 소진 → clarify
        trace["fallback_dispatch"] = {
            "action": "clarify",
            "index": next_index,
            "reason": "all intents exhausted",
        }
        return {
            "fallback_next": "clarify",
            "current_intent_index": next_index,
            "retry_count": 0,
            "trace": trace,
        }

    next_intent = ranking[next_index]

    # RAG 진입 시 language 분기
    if next_intent == "rag":
        fallback_next = "retrieve" if language_code == "ko" else "translate_ko"
    else:
        fallback_next = next_intent

    trace["fallback_dispatch"] = {
        "action": "fallback",
        "from_index": index,
        "to_index": next_index,
        "next_intent": next_intent,
        "fallback_next": fallback_next,
    }

    return {
        "fallback_next": fallback_next,
        "current_intent_index": next_index,
        "retry_count": 0,       # RAG 재시도 카운터 초기화
        "top_k": 5,             # RAG top_k 초기화
        "trace": trace,
    }


# ── fallback_router (conditional edge 함수) ──────────────────────────────────

def fallback_router(state: GraphState) -> str:
    """fallback_dispatch_node 이후 라우팅 결정.

    Returns:
        "done"         → tts_builder / END
        "retrieve"     → RAG (KO)
        "translate_ko" → RAG (Foreign)
        "smalltalk"    → smalltalk 노드
        "event"        → event 노드
        "struct_db"    → struct_db 노드
        "clarify"      → clarify 노드
    """
    return state.get("fallback_next", "clarify")
