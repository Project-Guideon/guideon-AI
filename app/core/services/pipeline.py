from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from app.core.services.llm_openai import OpenAILLM
from app.core.services.rag_pgvector import PgVectorRAG
from app.core.services.stt_google_v2 import GoogleSTTV2 as GoogleSTT
from app.core.services.tts_google import GoogleTTS
from app.graph.graph_builder import build_graph, build_text_graph
from app.core.services.rag_pgvector import OpenAIEmbedder
from app.core.language_profiles import get_profile


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
    answer_language: str = "ko"
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
        language_code: str = "ko",       # 하위 호환용; user_language가 우선
        user_language: str | None = None,
        answer_language: str | None = None,
        stt_language_code: str | None = None,
        mascot: Dict[str, Any] | None = None,
        device_id: str | None = None,
        chat_history: List[Dict[str, Any]] | None = None,
        daily_infos: List[Dict[str, Any]] | None = None,
        device_location: Dict[str, Any] | None = None,
    ) -> TextQAResult:
        lang2 = (user_language or language_code or "ko").split("-")[0].lower()
        ans_lang = (answer_language or lang2)
        stt_lang = stt_language_code or f"{lang2}-KR"  # 미지정 시 rough default

        initial_state: Dict[str, Any] = {
            "transcript": query,
            "language_code": lang2,
            "stt_language_code": stt_lang,
            "user_language": lang2,
            "answer_language": ans_lang,
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
            answer_language=result.get("answer_language") or result.get("user_language") or lang2,
            answer_found=result.get("check_result") == "good",
            contexts=result.get("retrieved_chunks", []),
            trace=result.get("trace", {}),
        )


class VoicePipeline:
    def __init__(self, stt: GoogleSTT, rag: PgVectorRAG, llm: OpenAILLM, tts: GoogleTTS):
        self.graph = build_graph(stt=stt, rag=rag, llm=llm, tts=tts)

    def run(
        self,
        audio_bytes: bytes,
        site_id: int = 1,
        language_code: str = "ko",
        mascot: Dict[str, Any] | None = None,
    ) -> VoiceQAResult:
        profile = get_profile(language_code)

        initial_state: Dict[str, Any] = {
            "audio": audio_bytes,
            "site_id": site_id,
            "stt_language_code": profile.stt_language_code,
            "user_language": profile.user_language,
            "answer_language": profile.answer_language,
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
