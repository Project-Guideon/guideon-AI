from __future__ import annotations

import os
from typing import Any, List, Optional

import requests

from app.graph.state import GraphState

CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8080")


def _fetch_by_category(
    site_id: Optional[int], device_id: str, category: Optional[str]
) -> List[dict]:
    """Core /places/nearby 단일 호출. 실패 시 [] 반환."""
    params: dict = {"siteId": site_id, "deviceId": device_id}
    if category:
        params["category"] = category

    safe_params = {k: v for k, v in params.items() if k != "deviceId"}
    print(f"[fetch_places] 요청: {CORE_BASE_URL}/internal/v1/places/nearby params={safe_params}")

    resp = requests.get(
        f"{CORE_BASE_URL}/internal/v1/places/nearby",
        params=params,
        timeout=5.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return [p for p in data if isinstance(p, dict)]
    if isinstance(data, dict) and isinstance(data.get("places"), list):
        return [p for p in data["places"] if isinstance(p, dict)]
    return []


def fetch_places_node(state: GraphState) -> dict:
    """Spring Boot Core 의 places/nearby API 를 호출해서 nearby_places 를 채우는 노드.

    intent_gate 가 struct_db 로 분류한 뒤 실행된다.
    - place_categories (top2, ex: ["TOILET", "INFO"]) 를 신뢰도 순으로 시도
    - top1 카테고리로 조회해 결과가 있으면 사용, 비어 있으면 top2 로 재조회
      (intent_gate 의 카테고리 오분류를 보정하기 위한 fallback)
    - 카테고리가 없으면(place_categories=[]) 전체 카테고리 거리순 조회
    - 실패 시 nearby_places=[] 로 struct_db_node 에 진입 → RAG fallback
    """
    site_id: Optional[int] = state.get("site_id")
    device_id: Optional[str] = state.get("device_id")
    place_categories: List[str] = state.get("place_categories") or []

    trace = dict(state.get("trace") or {})
    flow = list(trace.get("_flow") or [])
    flow.append("fetch_places")
    trace["_flow"] = flow

    if not device_id:
        trace["fetch_places"] = {"status": "no_device_id"}
        return {"nearby_places": [], "trace": trace}

    # 시도 순서: top1 → top2. 카테고리가 없으면 전체 조회(None) 1회.
    categories_to_try: List[Optional[str]] = list(place_categories) if place_categories else [None]

    nearby_places: List[dict] = []
    used_category: Optional[str] = None
    last_error: Optional[str] = None

    for category in categories_to_try:
        try:
            result = _fetch_by_category(site_id, device_id, category)
        except Exception as e:
            last_error = type(e).__name__
            print(f"[fetch_places] 오류: category={category}, {last_error}")
            continue  # 다음 카테고리로 fallback

        if result:
            nearby_places = result
            used_category = category
            break  # 결과 있으면 더 이상 fallback 안 함

    if nearby_places:
        print(f"[fetch_places] 결과: used_category={used_category}, "
              f"tried={place_categories}, count={len(nearby_places)}")
        for p in nearby_places:
            dist = p.get("distanceM")
            dist_text = f"{dist:.1f}m" if isinstance(dist, (int, float)) else "unknown"
            print(f"  - placeId={p.get('placeId')} [{p.get('name')}] "
                  f"category={p.get('category')} dist={dist_text} sameZone={p.get('sameZone')}")
        trace["fetch_places"] = {
            "status": "ok",
            "count": len(nearby_places),
            "categories": place_categories,
            "used_category": used_category,
        }
    else:
        # 모든 카테고리 조회 실패 또는 결과 0건
        status = "error" if last_error else "empty"
        print(f"[fetch_places] 결과 없음: status={status}, tried={place_categories}")
        trace["fetch_places"] = {
            "status": status,
            "count": 0,
            "categories": place_categories,
        }
        if last_error:
            trace["fetch_places"]["error"] = "core_places_failed"

    return {"nearby_places": nearby_places, "trace": trace}
