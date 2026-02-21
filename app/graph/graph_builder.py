from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.core.services.llm_openai import OpenAILLM
from app.core.services.rag_pgvector import PgVectorRAG
from app.core.services.stt_google import GoogleSTT
from app.core.services.tts_google import GoogleTTS

from app.graph.state import GraphState

# ── 노드 ──────────────────────────────────────────────────────────────────────
from app.graph.nodes.stt_node import make_stt_node
from app.graph.nodes.normalize_node import normalize_node
from app.graph.nodes.intent_gate import make_intent_gate_node
from app.graph.nodes.infotype_gate import make_infotype_gate_node
from app.graph.nodes.smalltalk_node import make_smalltalk_node
from app.graph.nodes.map_tool_node import map_tool_node
from app.graph.nodes.struct_db_node import struct_db_node
from app.graph.nodes.direct_llm_node import make_direct_llm_node
from app.graph.nodes.answer_compose import make_answer_compose_node
from app.graph.nodes.translate_node import make_translate_node
from app.graph.nodes.query_rewrite import make_query_rewrite_node
from app.graph.nodes.retrieve_node import make_retrieve_node
from app.graph.nodes.context_pack import context_pack_node
from app.graph.nodes.answer_generate import make_answer_generate_node
from app.graph.nodes.answer_check import answer_check_node
from app.graph.nodes.clarify_node import make_clarify_node
from app.graph.nodes.tts_builder import make_tts_builder_node

# ── 라우터 ────────────────────────────────────────────────────────────────────
from app.graph.routers.intent_router import intent_router
from app.graph.routers.infotype_router import infotype_router
from app.graph.routers.answer_check_router import answer_check_router


def build_graph(stt: GoogleSTT, rag: PgVectorRAG, llm: OpenAILLM, tts: GoogleTTS):
    """모든 노드와 엣지를 연결해서 LangGraph 실행 그래프를 빌드한다.

    앱 시작 시 1회 호출 → 반환된 graph 를 요청마다 invoke() 로 실행.
    """
    builder = StateGraph(GraphState)

    # ── 노드 등록 ─────────────────────────────────────────────────────
    # 서비스가 필요한 노드는 팩토리로 주입, 순수 로직 노드는 그대로 등록
    builder.add_node("stt",             make_stt_node(stt))
    builder.add_node("normalize",       normalize_node)
    builder.add_node("intent_gate",     make_intent_gate_node(llm))
    builder.add_node("infotype_gate",   make_infotype_gate_node(llm))
    builder.add_node("smalltalk",       make_smalltalk_node(llm))
    builder.add_node("map_tool",        map_tool_node)
    builder.add_node("struct_db",       struct_db_node)
    builder.add_node("direct_llm",      make_direct_llm_node(llm))
    builder.add_node("answer_compose",  make_answer_compose_node(llm))
    builder.add_node("translate_ko",    make_translate_node(llm))
    builder.add_node("query_rewrite",   make_query_rewrite_node(llm))
    builder.add_node("retrieve",        make_retrieve_node(rag))
    builder.add_node("context_pack",    context_pack_node)
    builder.add_node("answer_generate", make_answer_generate_node(llm))
    builder.add_node("answer_check",    answer_check_node)
    builder.add_node("clarify",         make_clarify_node(llm))
    builder.add_node("tts_builder",     make_tts_builder_node(tts))

    # ── 진입점 ────────────────────────────────────────────────────────
    builder.set_entry_point("stt")

    # ── 고정 엣지 (순서가 항상 동일한 구간) ──────────────────────────
    builder.add_edge("stt",            "normalize")
    builder.add_edge("normalize",      "intent_gate")

    # map_tool / struct_db → answer_compose → tts_builder
    builder.add_edge("map_tool",       "answer_compose")
    builder.add_edge("struct_db",      "answer_compose")
    builder.add_edge("answer_compose", "tts_builder")

    # smalltalk / direct_llm → tts_builder
    builder.add_edge("smalltalk",      "tts_builder")
    builder.add_edge("direct_llm",     "tts_builder")

    # Foreign RAG: translate_ko → query_rewrite
    # KO RAG:      infotype_router 가 직접 query_rewrite 로 보냄
    builder.add_edge("translate_ko",   "query_rewrite")

    # RAG 파이프라인 고정 구간
    builder.add_edge("query_rewrite",  "retrieve")
    builder.add_edge("retrieve",       "context_pack")
    builder.add_edge("context_pack",   "answer_generate")
    builder.add_edge("answer_generate","answer_check")

    # clarify → tts_builder → END
    builder.add_edge("clarify",        "tts_builder")
    builder.add_edge("tts_builder",    END)

    # ── 조건 엣지 (분기 구간) ─────────────────────────────────────────
    builder.add_conditional_edges(
        "intent_gate",
        intent_router,
        {
            "smalltalk":    "smalltalk",
            "info_request": "infotype_gate",
        },
    )

    builder.add_conditional_edges(
        "infotype_gate",
        infotype_router,
        {
            "map_tool":      "map_tool",
            "struct_db":     "struct_db",
            "direct_llm":    "direct_llm",
            "query_rewrite": "query_rewrite",   # KO RAG 경로
            "translate_ko":  "translate_ko",    # Foreign RAG 경로
        },
    )

    builder.add_conditional_edges(
        "answer_check",
        answer_check_router,
        {
            "good":  "tts_builder",  # 품질 통과 → TTS
            "retry": "retrieve",     # 재검색 루프백 (MMR 적용)
            "bad":   "clarify",      # 최대 재시도 초과 → clarifying question
        },
    )

    return builder.compile()
