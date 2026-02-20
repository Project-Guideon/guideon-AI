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
        query = self.stt.transcribe(audio_bytes)#음성데이터(bytes)를 텍스트로 변환

        if not query:
            answer = "죄송해요. 음성을 인식하지 못했어요. 다시 말씀해 주세요."
            voice = self.tts.synthesize(answer) #텍스트를 음성데이터(bytes)로 변환
            return VoiceQAResult(query="", answer=answer, voice_bytes=voice, contexts=[])

        print(f"\n[STT] 인식된 텍스트: {query}")

        #아랫부분 langraph호출해서 처리로 바꾸기
        contexts = self.rag.retrieve(query=query, site_id=site_id, k=k)
        print(f"[RAG] 검색된 청크 수: {len(contexts)}")
        for i, c in enumerate(contexts):
            print(f"  [{i+1}] similarity={c.similarity:.4f} | {c.content[:60]}...")

        answer = self.llm.generate(query=query, contexts=contexts)
        print(f"[LLM] 생성된 답변: {answer}")

        voice = self.tts.synthesize(answer)
        print(f"[TTS] 음성 생성 완료: {len(voice)} bytes\n")

        return VoiceQAResult(query=query, answer=answer, voice_bytes=voice, contexts=contexts)
