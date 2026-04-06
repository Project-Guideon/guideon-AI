from __future__ import annotations

from app.graph.state import GraphState


def infotype_router(state: GraphState) -> str:
    """infotype_gate 결과 + language_code 로 다음 노드를 결정.

    KO/Foreign 분기가 여기서 실질적으로 처리됨.

    Returns:
        "map_tool"      → map_tool 노드
        "struct_db"     → struct_db 노드
        "direct_llm"    → direct_llm 노드
        "query_rewrite" → KO RAG 경로 (translate 없이 바로)
        "translate_ko"  → Foreign RAG 경로 (한국어 변환 먼저)
    """
    info_type: str = state.get("info_type", "rag")
    language_code: str = state.get("language_code", "ko")

    if info_type == "rag":
        return "query_rewrite" if language_code == "ko" else "translate_ko"

    return info_type  # "map_tool" | "struct_db" | "direct_llm"
