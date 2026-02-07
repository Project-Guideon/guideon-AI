# services/voice_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from app.core.services.rag_pgvector import RetrievedChunk, PgVectorRAG
from app.core.services.stt_google import GoogleSTT
from app.core.services.tts_google import GoogleTTS
from app.core.services.llm_openai import OpenAILLM


@dataclass
class VoiceQAResult:
    query: str
    answer: str
    voice_bytes: bytes
    contexts: List[RetrievedChunk]


class VoicePipeline:
    def __init__(self, stt: GoogleSTT, rag: PgVectorRAG, llm: OpenAILLM, tts: GoogleTTS):
        self.stt = stt
        self.rag = rag
        self.llm = llm
        self.tts = tts

    def run(self, audio_bytes: bytes, site_id: int = 1, k: int = 5) -> VoiceQAResult:
        query = self.stt.transcribe(audio_bytes)

        if not query:
            answer = "죄송해요. 음성을 인식하지 못했어요. 다시 말씀해 주세요."
            voice = self.tts.synthesize(answer)
            return VoiceQAResult(query="", answer=answer, voice_bytes=voice, contexts=[])

        contexts = self.rag.retrieve(query=query, site_id=site_id, k=k)
        answer = self.llm.generate(query=query, contexts=contexts)
        voice = self.tts.synthesize(answer)

        return VoiceQAResult(query=query, answer=answer, voice_bytes=voice, contexts=contexts)
