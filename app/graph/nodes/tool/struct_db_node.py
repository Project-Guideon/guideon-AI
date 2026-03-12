from __future__ import annotations

from app.graph.state import GraphState

# TODO: 실제 장소 DB 테이블 연결 시 이 노드 내부만 교체


def struct_db_node(state: GraphState) -> dict:
    """장소/위치 조회 노드 (현재 stub).

    실제 구현 시 교체할 것:
    - 입력: normalized_text, site_id, user_language
    - 조회: DB에 저장된 장소 (화장실, 주차장, 건물, 문, 매표소, 식당 등)의 위치
    - 출력: answer_text (user_language 로 LLM 직접 생성)
    """
    site_id: int = state.get("site_id", 1)
    text: str = state.get("normalized_text", "")

    # ── stub: 데이터 없음 → check_result = "bad" ─────────────────────
    trace = dict(state.get("trace") or {})
    flow = list(trace.get("_flow") or [])
    flow.append("struct_db")
    trace["_flow"] = flow
    trace["struct_db"] = {"status": "stub", "query": text, "site_id": site_id}

    return {
        "answer_text": "",
        "check_result": "bad",
        "trace": trace,
    }
