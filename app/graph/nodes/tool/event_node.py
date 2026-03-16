from __future__ import annotations

from app.graph.state import GraphState

# TODO: 실제 이벤트 DB 테이블 생성 후 이 노드 내부만 교체


def event_node(state: GraphState) -> dict:
    """이벤트 조회 노드 (현재 stub).

    실제 구현 시 교체할 것:
    - 입력: normalized_text, site_id, user_language
    - 조회: 현재/예정 이벤트, 축제, 공연, 전시 등
    - 출력: answer_text (user_language 로 LLM 직접 생성)
    """
    site_id: int = state.get("site_id", 1)
    text: str = state.get("normalized_text", "")

    # ── stub: 데이터 없음 → check_result = "bad" ─────────────────────
    trace = dict(state.get("trace") or {})
    flow = list(trace.get("_flow") or [])
    flow.append("event")
    trace["_flow"] = flow
    trace["event"] = {"status": "stub", "query": text, "site_id": site_id}

    return {
        "answer_text": "",
        "check_result": "bad",
        "trace": trace,
    }
