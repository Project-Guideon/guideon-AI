from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.core.services.llm_openai import OpenAILLM
from app.core.services.rag_pgvector import PgVectorRAG
from app.core.services.rag_pgvector_v2 import PgVectorRAG_V2
from app.core.services.stt_google import GoogleSTT
from app.core.services.tts_google import GoogleTTS

from app.graph.state import GraphState

# ── 노드: input ───────────────────────────────────────────────────────────────
from app.graph.nodes.input.stt_node import make_stt_node
from app.graph.nodes.input.normalize_node import normalize_node

# ── 노드: classify ────────────────────────────────────────────────────────────
from app.graph.nodes.classify.intent_gate import make_intent_gate_node

# ── 노드: rag ─────────────────────────────────────────────────────────────────
from app.graph.nodes.rag.translate_node import make_translate_node
from app.graph.nodes.rag.retrieve_node import make_retrieve_node
from app.graph.nodes.rag.retrieve_node_v2 import make_retrieve_node_v2
from app.graph.nodes.rag.context_pack import context_pack_node

# ── 노드: answer ──────────────────────────────────────────────────────────────
from app.graph.nodes.answer.answer_generate import make_answer_generate_node
from app.graph.nodes.answer.answer_check import answer_check_node
from app.graph.nodes.answer.clarify_node import make_clarify_node
from app.graph.nodes.answer.smalltalk_node import make_smalltalk_node

# ── 노드: tool ────────────────────────────────────────────────────────────────
from app.graph.nodes.tool.fetch_places_node import fetch_places_node
from app.graph.nodes.tool.struct_db_node import make_struct_db_node
from app.graph.nodes.tool.event_node import event_node

# ── 노드: output ──────────────────────────────────────────────────────────────
from app.graph.nodes.output.tts_builder import make_tts_builder_node

# ── 라우터 ────────────────────────────────────────────────────────────────────
from app.graph.routers.intent_router import intent_router
from app.graph.routers.answer_check_router import answer_check_router
from app.graph.routers.fallback_router import fallback_dispatch_node, fallback_router


# ── 공통 노드/엣지 등록 (text_graph, full_graph 공유) ─────────────────────────

def _register_core_nodes(builder: StateGraph, rag: PgVectorRAG, llm: OpenAILLM):
    """STT/TTS 를 제외한 공통 노드를 등록한다."""
    builder.add_node("normalize",          normalize_node)
    builder.add_node("intent_gate",        make_intent_gate_node(llm))
    builder.add_node("smalltalk",          make_smalltalk_node(llm))
    builder.add_node("event",              event_node)
    builder.add_node("fetch_places",       fetch_places_node)
    builder.add_node("struct_db",          make_struct_db_node(llm))
    builder.add_node("translate_ko",       make_translate_node(llm))

    _retrieve = make_retrieve_node_v2(rag) if isinstance(rag, PgVectorRAG_V2) else make_retrieve_node(rag)
    builder.add_node("retrieve",           _retrieve)
    builder.add_node("context_pack",       context_pack_node)
    builder.add_node("answer_generate",    make_answer_generate_node(llm))
    builder.add_node("answer_check",       answer_check_node)
    builder.add_node("fallback_dispatch",  fallback_dispatch_node)
    builder.add_node("clarify",            make_clarify_node(llm))


def _register_core_edges(builder: StateGraph, end_node: str):
    """공통 엣지를 등록한다.

    end_node: build_text_graph 에서는 END, build_graph 에서는 "tts_builder"
    """
    builder.add_edge("normalize", "intent_gate")

    # ── 비-RAG 분기 → fallback_dispatch 수렴 ──────────────────────────
    builder.add_edge("smalltalk",      "fallback_dispatch")
    builder.add_edge("event",          "fallback_dispatch")
    builder.add_edge("fetch_places",   "struct_db")
    builder.add_edge("struct_db",      "fallback_dispatch")

    # ── RAG 파이프라인 고정 구간 ──────────────────────────────────────
    builder.add_edge("translate_ko",    "retrieve")
    builder.add_edge("retrieve",        "context_pack")
    builder.add_edge("context_pack",    "answer_generate")
    builder.add_edge("answer_generate", "answer_check")

    # ── clarify → 종료 ───────────────────────────────────────────────
    builder.add_edge("clarify", end_node)

    # ── 조건 엣지: intent_gate → 4분기 ───────────────────────────────
    builder.add_conditional_edges(
        "intent_gate",
        intent_router,
        {
            "smalltalk":    "smalltalk",
            "event":        "event",
            "struct_db":    "fetch_places",  # DB 조회 후 struct_db 로
            "retrieve":     "retrieve",      # RAG (KO)
            "translate_ko": "translate_ko",  # RAG (Foreign)
        },
    )

    # ── 조건 엣지: answer_check → retry / fallback_dispatch ──────────
    builder.add_conditional_edges(
        "answer_check",
        answer_check_router,
        {
            "retry":             "retrieve",
            "fallback_dispatch": "fallback_dispatch",
        },
    )

    # ── 조건 엣지: fallback_dispatch → 재라우팅 / 종료 ───────────────
    builder.add_conditional_edges(
        "fallback_dispatch",
        fallback_router,
        {
            "done":         end_node,
            "smalltalk":    "smalltalk",
            "event":        "event",
            "struct_db":    "fetch_places",  # fallback 시에도 DB 재조회
            "retrieve":     "retrieve",
            "translate_ko": "translate_ko",
            "clarify":      "clarify",
        },
    )


# ── 빌더 함수 ────────────────────────────────────────────────────────────────

def build_text_graph(rag: PgVectorRAG, llm: OpenAILLM):
    """STT/TTS 없이 텍스트 입출력만 처리하는 LangGraph 그래프."""
    builder = StateGraph(GraphState)

    _register_core_nodes(builder, rag, llm)

    builder.set_entry_point("normalize")

    _register_core_edges(builder, end_node=END)

    return builder.compile()


def build_graph(stt: GoogleSTT, rag: PgVectorRAG, llm: OpenAILLM, tts: GoogleTTS):
    """모든 노드와 엣지를 연결해서 LangGraph 실행 그래프를 빌드한다.

    앱 시작 시 1회 호출 → 반환된 graph 를 요청마다 invoke() 로 실행.
    """
    builder = StateGraph(GraphState)

    # ── STT / TTS 노드 추가 등록 ──────────────────────────────────────
    builder.add_node("stt",         make_stt_node(stt))
    builder.add_node("tts_builder", make_tts_builder_node(tts))

    _register_core_nodes(builder, rag, llm)

    # ── 진입점 ────────────────────────────────────────────────────────
    builder.set_entry_point("stt")
    builder.add_edge("stt", "normalize")

    _register_core_edges(builder, end_node="tts_builder")

    builder.add_edge("tts_builder", END)

    return builder.compile()
