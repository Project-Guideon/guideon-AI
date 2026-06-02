from __future__ import annotations

import os
from typing import Any, List, Optional

import requests
from langsmith import traceable

from app.graph.state import GraphState
from app.graph.nodes.tool.navigation_node import _match_from_nearby

CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8080")


@traceable(name="fetch_places_category_fallback", run_type="tool")
def _log_category_fallback(
    from_category: Optional[str],
    to_category: Optional[str],
    reason: str,
) -> dict:
    """카테고리 fallback 발생 시 Langsmith에 스팬으로 기록합니다."""
    print(f"[fetch_places] fallback: {from_category} → {to_category} (reason={reason})")
    return {"from": from_category, "to": to_category, "reason": reason}


@traceable(name="fetch_places_by_category", run_type="tool")
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


@traceable(name="fetch_places", run_type="tool")
def fetch_places_node(state: GraphState) -> dict:
    """Spring Boot Core 의 places/nearby API 를 호출해서 nearby_places 를 채우는 노드.

    intent_gate 가 struct_db 로 분류한 뒤 실행된다.

    [카테고리 fallback 기준]
    - place_name_query 있음 (예: "샐러박스 어디야"):
        카테고리 결과 안에 그 장소가 매칭되는지로 fallback 판단.
        SHOP에서 9개 나와도 샐러박스가 없으면 OTHER 로 재조회.
    - place_name_query 없음 (예: "화장실 어디야"):
        결과 건수 기준. top1 결과 있으면 즉시 사용.

    - 카테고리가 없으면(place_categories=[]) 전체 카테고리 거리순 조회
    - 실패 시 nearby_places=[] 로 struct_db_node 에 진입 → RAG fallback
    """
    site_id: Optional[int] = state.get("site_id")
    device_id: Optional[str] = state.get("device_id")
    place_categories: List[str] = state.get("place_categories") or []
    raw_place_name_query = state.get("place_name_query")
    place_name_query: Optional[str] = (
        raw_place_name_query.strip()
        if isinstance(raw_place_name_query, str) and raw_place_name_query.strip()
        else None
    )

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
    name_matched = False

    for idx, category in enumerate(categories_to_try):
        try:
            result = _fetch_by_category(site_id, device_id, category)
        except Exception as e:
            last_error = type(e).__name__
            print(f"[fetch_places] 오류: category={category}, {last_error}")
            if idx > 0:
                _log_category_fallback(
                    from_category=categories_to_try[idx - 1],
                    to_category=category,
                    reason="api_error",
                )
            continue

        if not result:
            # 결과 0건 → 다음 카테고리 fallback
            if idx + 1 < len(categories_to_try):
                _log_category_fallback(
                    from_category=category,
                    to_category=categories_to_try[idx + 1],
                    reason="empty_result",
                )
            continue

        # 특정 장소 검색: 결과 안에 그 장소가 있는지로 fallback 판단
        if place_name_query:
            if _match_from_nearby(place_name_query, result):
                nearby_places = result
                used_category = category
                name_matched = True
                break
            # 결과는 있지만 찾는 장소가 없음 → 첫 결과는 보관(LLM fallback용) 후 다음 카테고리
            if not nearby_places:
                nearby_places = result
                used_category = category
            if idx + 1 < len(categories_to_try):
                _log_category_fallback(
                    from_category=category,
                    to_category=categories_to_try[idx + 1],
                    reason="name_not_matched",
                )
            continue

        # 카테고리 검색: 결과 있으면 즉시 사용
        nearby_places = result
        used_category = category
        break

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
            "name_query": place_name_query,
            "name_matched": name_matched,
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
