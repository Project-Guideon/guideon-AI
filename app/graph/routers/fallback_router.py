from __future__ import annotations

from typing import List

from app.graph.state import GraphState

ALLOWED_FALLBACK_NEXT = {"done", "smalltalk", "event", "struct_db", "retrieve", "translate_ko", "clarify"}


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
    flow = list(trace.get("_flow") or [])
    flow.append("fallback_dispatch")
    trace["_flow"] = flow

    # fallback 이력을 리스트로 누적 (여러 번 호출되므로 덮어쓰지 않음)
    fallback_history = list(trace.get("fallback_history") or [])

    if check_result == "good":
        entry = {"action": "done", "index": index, "intent": ranking[index] if index < len(ranking) else "?"}
        fallback_history.append(entry)
        trace["fallback_history"] = fallback_history
        return {"fallback_next": "done", "trace": trace}

    # ── 다음 의도로 이동 ──────────────────────────────────────────────
    next_index = index + 1

    if next_index >= len(ranking):
        # 모든 의도 소진 → clarify
        entry = {
            "action": "clarify",
            "index": next_index,
            "reason": "all intents exhausted",
        }
        fallback_history.append(entry)
        trace["fallback_history"] = fallback_history
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
    elif next_intent in {"smalltalk", "event", "struct_db"}:
        fallback_next = next_intent
    else:
        fallback_next = "clarify"

    entry = {
        "action": "fallback",
        "from_index": index,
        "from_intent": ranking[index] if index < len(ranking) else "?",
        "to_index": next_index,
        "to_intent": next_intent,
        "fallback_next": fallback_next,
    }
    fallback_history.append(entry)
    trace["fallback_history"] = fallback_history

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
    nxt = state.get("fallback_next", "clarify")
    return nxt if nxt in ALLOWED_FALLBACK_NEXT else "clarify"
