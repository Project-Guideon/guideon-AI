from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.services.llm_openai import OpenAILLM
from app.core.services.rag_pgvector import PgVectorRAG
from app.core.services.stt_google import GoogleSTT
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
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


class TextPipeline:
    def __init__(self, rag: PgVectorRAG, llm: OpenAILLM):
        self.graph = build_text_graph(rag=rag, llm=llm)

    def run(self, query: str, site_id: int = 1, language_code: str = "ko") -> TextQAResult:
        lang2 = language_code.split("-")[0].lower()  # "ko-KR" → "ko"
        initial_state = {
            "transcript": query,        # STT 결과 대신 텍스트를 직접 주입
            "language_code": lang2,
            "user_language": lang2,
            "site_id": site_id,
            "top_k": 5,
            "retry_count": 0,
            "trace": {},
        }
        result = self.graph.invoke(initial_state)
        return TextQAResult(
            query=query,
            answer=result.get("answer_text", ""),
            contexts=result.get("retrieved_chunks", []),
            trace=result.get("trace", {}),
        )


class VoicePipeline:
    def __init__(self, stt: GoogleSTT, rag: PgVectorRAG, llm: OpenAILLM, tts: GoogleTTS):
        # 서비스를 직접 보관하는 대신 LangGraph 그래프로 조립
        self.graph = build_graph(stt=stt, rag=rag, llm=llm, tts=tts)

    def run(self, audio_bytes: bytes, site_id: int = 1) -> VoiceQAResult:
        # top_k / retry_count 는 그래프 내부(answer_check)에서 동적으로 관리
        initial_state = {
            "audio": audio_bytes,
            "site_id": site_id,
            "top_k": 5,
            "retry_count": 0,
            "trace": {},
        }

        result = self.graph.invoke(initial_state)

        return VoiceQAResult(
            query=result.get("transcript", ""),
            answer=result.get("answer_text", ""),
            voice_bytes=result.get("tts_audio", b""),
            contexts=result.get("retrieved_chunks", []),
            trace=result.get("trace", {}),
        )

# 실시간 tts용
class StreamingVoicePipeline:
    def __init__(self, stt: GoogleSTT, rag: PgVectorRAG, llm: OpenAILLM, tts: GoogleTTS):
        self.stt = stt
        self.rag = rag
        self.llm = llm
        self.tts = tts
        self.text_graph = build_text_graph(rag=rag, llm=llm)

    def run_stt(self, audio_bytes: bytes) -> dict:
        stt_result = self.stt.transcribe(audio_bytes)
        return {
            "transcript": stt_result.transcript,
            "language_code": getattr(stt_result, "language_code", "ko"),
            "confidence": getattr(stt_result, "confidence", 0.0),
        }

    def generate_answer(self, transcript: str, site_id: int = 1, language_code: str = "ko") -> dict:
        lang2 = language_code.split("-")[0].lower()

        initial_state = {
            "transcript": transcript,
            "language_code": lang2,
            "user_language": lang2,
            "site_id": site_id,
            "top_k": 5,
            "retry_count": 0,
            "trace": {},
        }

        result = self.text_graph.invoke(initial_state)
        return {
            "answer_text": result.get("answer_text", ""),
            "contexts": result.get("retrieved_chunks", []),
            "trace": result.get("trace", {}),
            "language_code": lang2,
        }

    def split_sentences(self, answer_text: str) -> list[str]:
        import re
        sentence_split_re = re.compile(r'(?<=[.!?])\s+|(?<=다\.)\s*|(?<=요\.)\s*|(?<=니다\.)\s*')
        return [s.strip() for s in sentence_split_re.split(answer_text) if s and s.strip()]

    def synthesize_sentence(self, sentence: str) -> bytes:
        return self.tts.synthesize(sentence)
