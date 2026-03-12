from __future__ import annotations

from app.graph.state import GraphState

_MAX_CONTEXT_CHUNKS = 5   # answer_generate 에 넘길 최대 청크 수


def context_pack_node(state: GraphState) -> dict:
    """검색된 청크를 정리해 answer_generate 에 넘길 컨텍스트를 준비.

    - 앞 100자 기준 중복 청크 제거
    - similarity 내림차순 정렬
    - 상위 MAX_CONTEXT_CHUNKS 개로 압축
    """
    chunks: list = list(state.get("retrieved_chunks") or [])

    # 1. 앞 100자 기준 중복 제거 (같은 문서의 인접 청크가 중복 선택될 때 방지)
    seen: set = set()
    unique: list = []
    for c in chunks:
        key = c["content"][:100].strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # 2. similarity 내림차순 정렬
    unique.sort(key=lambda c: c.get("similarity", 0.0), reverse=True)

    # 3. 상위 N개로 압축
    packed = unique[:_MAX_CONTEXT_CHUNKS]

    trace = dict(state.get("trace") or {})
    trace["context_pack"] = {
        "before": len(chunks),
        "after": len(packed),
        "top_similarity": round(packed[0]["similarity"], 4) if packed else 0.0,
    }

    return {"retrieved_chunks": packed, "trace": trace}
