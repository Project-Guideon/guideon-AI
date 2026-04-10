"""
공유 인스턴스 모음.
기존 main_fastapi.py 상단 전역 초기화 코드를 그대로 분리.
각 라우터(api/*.py)에서 여기서 import해서 사용.
"""
import os

from openai import OpenAI
from langsmith import traceable

from app.core.services.stt_google import GoogleSTT, STTConfig
from app.core.services.tts_google import GoogleTTS, TTSConfig
from app.core.services.rag_pgvector import PgVectorRAG, OpenAIEmbedder
from app.core.services.rag_pgvector_v2 import PgVectorRAG_V2
from app.core.services.llm_openai import OpenAILLM, LLMConfig
from app.core.services.pipeline import VoicePipeline, TextPipeline, StreamingVoicePipeline

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

stt = GoogleSTT(STTConfig(primary_language="ko-KR", sample_rate_hz=16000))
tts = GoogleTTS(TTSConfig(language_code="ko-KR"))
embedder = OpenAIEmbedder(client=client, model="text-embedding-3-small")

RAG_VERSION = os.getenv("RAG_VERSION", "v1")
if RAG_VERSION == "v2":
    rag = PgVectorRAG_V2(embedder=embedder)
else:
    rag = PgVectorRAG(embedder=embedder)

llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", temperature=0.2, max_tokens=150))

pipeline = VoicePipeline(stt=stt, rag=rag, llm=llm, tts=tts)
text_pipeline = TextPipeline(rag=rag, llm=llm)
streaming_pipeline = StreamingVoicePipeline(stt=stt, rag=rag, llm=llm, tts=tts)

@traceable(name="voice_qa_pipeline")
def traced_voice_run(audio_bytes: bytes, site_id: int, mascot: dict | None = None):
    return pipeline.run(audio_bytes, site_id=site_id, mascot=mascot)


@traceable(name="text_qa_pipeline")
def traced_text_run(query: str, site_id: int, language_code: str, mascot: dict | None = None):
    return text_pipeline.run(query=query, site_id=site_id, language_code=language_code, mascot=mascot)


@traceable(name="internal_qa_pipeline")
def traced_internal_run(initial_state: dict) -> dict:
    return text_pipeline.graph.invoke(initial_state)
