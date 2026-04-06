from __future__ import annotations

from app.graph.state import GraphState

# TODO: 실제 지도 API / 건물 내부 POI DB 연결 시 이 노드 내부만 교체


def map_tool_node(state: GraphState) -> dict:
    """지도·위치 조회 노드 (현재 stub).

    실제 구현 시 교체할 것:
    - 입력: normalized_text, site_id
    - 출력: POI 결과 (이름, 위치 설명, 층/건물/좌표 등)
    결과는 answer_compose 에서 자연어로 변환됨.
    """
    site_id: int = state.get("site_id", 1)
    text: str = state.get("normalized_text", "")

    # ── stub 결과 ────────────────────────────────────────────────────
    poi_result = {
        "status": "stub",
        "query": text,
        "site_id": site_id,
        "result": None,   # 실제 API 연결 시 POI 데이터로 교체
    }

    trace = dict(state.get("trace") or {})
    trace["map_tool"] = {"status": "stub", "query": text}

    return {"poi_result": poi_result, "trace": trace}
