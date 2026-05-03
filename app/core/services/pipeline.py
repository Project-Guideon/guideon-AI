from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from app.core.services.llm_openai import OpenAILLM
from app.core.services.rag_pgvector import PgVectorRAG
# from app.core.services.stt_google import GoogleSTT
from app.core.services.stt_google_v2 import GoogleSTTV2 as GoogleSTT
from app.core.services.tts_google import GoogleTTS
from app.graph.graph_builder import build_graph, build_text_graph
from app.core.services.rag_pgvector import OpenAIEmbedder

@dataclass
class VoiceQAResult:
    query: str
    answer: str
    voice_bytes: bytes
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextQAResult:
    query: str
    answer: str
    category: str = "GENERAL"
    answer_found: bool = False
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


class TextPipeline:
    def __init__(self, rag: PgVectorRAG, llm: OpenAILLM):
        self.graph = build_text_graph(rag=rag, llm=llm)

    def run(
        self,
        query: str,
        site_id: int = 1,
        language_code: str = "ko",
        mascot: Dict[str, Any] | None = None,
        device_id: str | None = None,
        chat_history: List[Dict[str, Any]] | None = None,
        daily_infos: List[Dict[str, Any]] | None = None,
        device_location: Dict[str, Any] | None = None,
    ) -> TextQAResult:
        lang2 = language_code.split("-")[0].lower()  # "ko-KR" → "ko"
        initial_state: Dict[str, Any] = {
            "transcript": query,        # STT 결과 대신 텍스트를 직접 주입
            "language_code": lang2,
            "detected_language_code": lang2,
            "user_language": lang2,
            "site_id": site_id,
            "device_id": device_id,
            "system_prompt": "",
            "mascot_name": "",
            "mascot_greeting": "",
            "mascot_base_persona": "",
            "mascot_smalltalk_style": "",
            "mascot_struct_db_style": "",
            "mascot_RAG_style": "",
            "mascot_event_style": "",
            "top_k": 5,
            "retry_count": 0,
            "chat_history": chat_history or [],
            "nearby_places": [],
            "daily_infos": daily_infos or [],
            "device_location": device_location or {},
            "place_category": None,
            "place_id": None,
            "emotion": "",
            "category": "",
            "trace": {},
        }
        if mascot:
            initial_state.update(mascot)
        result = self.graph.invoke(initial_state)
        return TextQAResult(
            query=query,
            answer=result.get("answer_text", ""),
            category=result.get("category") or "GENERAL",
            answer_found=result.get("check_result") == "good",
            contexts=result.get("retrieved_chunks", []),
            trace=result.get("trace", {}),
        )


class VoicePipeline:
    def __init__(self, stt: GoogleSTT, rag: PgVectorRAG, llm: OpenAILLM, tts: GoogleTTS):
        # 서비스를 직접 보관하는 대신 LangGraph 그래프로 조립
        self.graph = build_graph(stt=stt, rag=rag, llm=llm, tts=tts)

    def run(
        self,
        audio_bytes: bytes,
        site_id: int = 1,
        mascot: Dict[str, Any] | None = None,
    ) -> VoiceQAResult:
        # top_k / retry_count 는 그래프 내부(answer_check)에서 동적으로 관리
        initial_state: Dict[str, Any] = {
            "audio": audio_bytes,
            "site_id": site_id,
            "top_k": 5,
            "retry_count": 0,
            "trace": {},
        }
        if mascot:
            initial_state.update(mascot)

        result = self.graph.invoke(initial_state)

        return VoiceQAResult(
            query=result.get("transcript", ""),
            answer=result.get("answer_text", ""),
            voice_bytes=result.get("tts_audio", b""),
            contexts=result.get("retrieved_chunks", []),
            trace=result.get("trace", {}),
        )

