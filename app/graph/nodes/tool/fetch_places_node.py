from __future__ import annotations

import os
from typing import Optional

import requests

from app.graph.state import GraphState

CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8080")


def fetch_places_node(state: GraphState) -> dict:
    """Spring Boot Core 의 places/nearby API 를 호출해서 nearby_places 를 채우는 노드.

    intent_gate 가 struct_db 로 분류한 뒤 실행된다.
    - place_category (ex: RESTROOM) 를 필터로 넘겨 해당 카테고리 장소만 거리순 조회
    - 실패 시 nearby_places=[] 로 struct_db_node 에 진입 → RAG fallback
    """
    site_id: Optional[int] = state.get("site_id")
    device_id: Optional[str] = state.get("device_id")
    place_category: Optional[str] = state.get("place_category")

    trace = dict(state.get("trace") or {})
    flow = list(trace.get("_flow") or [])
    flow.append("fetch_places")
    trace["_flow"] = flow

    if not device_id:
        trace["fetch_places"] = {"status": "no_device_id"}
        return {"nearby_places": [], "trace": trace}

    try:
        params: dict = {"siteId": site_id, "deviceId": device_id}
        if place_category:
            params["category"] = place_category

        print(f"[fetch_places] 요청: {CORE_BASE_URL}/internal/v1/places/nearby params={params}")

        resp = requests.get(
            f"{CORE_BASE_URL}/internal/v1/places/nearby",
            params=params,
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        nearby_places = data if isinstance(data, list) else data.get("places", [])

        print(f"[fetch_places] 결과: category={place_category}, count={len(nearby_places)}")
        for p in nearby_places:
            print(f"  - placeId={p.get('placeId')} [{p.get('name')}] category={p.get('category')} dist={p.get('distanceM'):.1f}m sameZone={p.get('sameZone')}")

        trace["fetch_places"] = {
            "status": "ok",
            "count": len(nearby_places),
            "category": place_category,
        }
        return {"nearby_places": nearby_places, "trace": trace}

    except Exception as e:
        print(f"[fetch_places] 오류: {e}")
        trace["fetch_places"] = {"status": "error", "error": str(e)}
        return {"nearby_places": [], "trace": trace}
