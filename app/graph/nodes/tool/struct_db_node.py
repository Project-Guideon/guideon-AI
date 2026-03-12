from __future__ import annotations

from app.graph.state import GraphState

# TODO: 실제 운영시간/요금/예약 등 정형 DB 연결 시 이 노드 내부만 교체


def struct_db_node(state: GraphState) -> dict:
    """정형 정보 조회 노드 (현재 stub).

    실제 구현 시 교체할 것:
    - 입력: normalized_text, site_id
    - 조회 키: 운영시간 / 입장료 / 휴무 / 예약 / 연락처 / 규정 등
    결과는 answer_compose 에서 자연어로 변환됨.
    """
    site_id: int = state.get("site_id", 1)
    text: str = state.get("normalized_text", "")

    # ── stub 결과 ────────────────────────────────────────────────────
    db_result = {
        "status": "stub",
        "query": text,
        "site_id": site_id,
        "result": None,   # 실제 DB 조회 시 정형 데이터로 교체
    }

    trace = dict(state.get("trace") or {})
    trace["struct_db"] = {"status": "stub", "query": text}

    return {"db_result": db_result, "trace": trace}
