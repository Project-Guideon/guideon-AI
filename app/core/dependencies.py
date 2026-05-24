"""
공유 인스턴스 모음.
기존 main_fastapi.py 상단 전역 초기화 코드를 그대로 분리.
각 라우터(api/*.py)에서 여기서 import해서 사용.
"""
import logging
import os

from openai import OpenAI
from langsmith import traceable

from app.core.services.realtime_stt import OpenAIRealtimeSTT as _STTClass, RealtimeSTTConfig as _STTConfig
from app.core.services.tts_google import GoogleTTS, TTSConfig
from app.core.services.tts_cartesia import CartesiaTTS
from app.core.services.rag_pgvector import PgVectorRAG, OpenAIEmbedder
from app.core.services.rag_pgvector_v2 import PgVectorRAG_V2
from app.core.services.llm_openai import OpenAILLM, LLMConfig
from app.core.services.pipeline import VoicePipeline, TextPipeline

_logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


stt = _STTClass(_STTConfig())
# Google TTS는 Cartesia 실패 시 폴백으로 계속 유지
tts = GoogleTTS(TTSConfig(language_code="ko-KR"))

# Cartesia TTS — CARTESIA_API_KEY만 있으면 활성화
# voice_id는 Kiosk BFF가 start 메시지로 전달 (tb_mascot.tts_voice_id)
_cartesia_api_key = os.getenv("CARTESIA_API_KEY")
cartesia_tts: CartesiaTTS | None = None
if _cartesia_api_key:
    try:
        cartesia_tts = CartesiaTTS(api_key=_cartesia_api_key)
        _logger.info("CartesiaTTS 초기화 완료 (voice_id는 마스코트별 지정)")
    except Exception as _exc:
        _logger.warning("CartesiaTTS 초기화 실패 → Google TTS 폴백 사용: %s", _exc)

embedder = OpenAIEmbedder(client=client, model="text-embedding-3-small")

RAG_VERSION = os.getenv("RAG_VERSION", "v1")
if RAG_VERSION == "v2":
    rag = PgVectorRAG_V2(embedder=embedder)
else:
    rag = PgVectorRAG(embedder=embedder)

llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", temperature=0.2, max_tokens=150))

pipeline = VoicePipeline(stt=stt, rag=rag, llm=llm, tts=tts)
text_pipeline = TextPipeline(rag=rag, llm=llm)

@traceable(name="voice_qa_pipeline")
def traced_voice_run(audio_bytes: bytes, site_id: int, language_code: str = "ko", mascot: dict | None = None):
    return pipeline.run(audio_bytes, site_id=site_id, language_code=language_code, mascot=mascot)


@traceable(name="text_qa_pipeline")
def traced_text_run(
    query: str,
    site_id: int,
    *,
    user_language: str = "ko",
    answer_language: str = "ko",
    stt_language_code: str = "ko-KR",
    mascot: dict | None = None,
    device_id: str | None = None,
    chat_history: list[dict] | None = None,
    daily_infos: list[dict] | None = None,
    device_location: dict | None = None,
):
    return text_pipeline.run(
        query=query,
        site_id=site_id,
        user_language=user_language,
        answer_language=answer_language,
        stt_language_code=stt_language_code,
        mascot=mascot,
        device_id=device_id,
        chat_history=chat_history,
        daily_infos=daily_infos,
        device_location=device_location,
    )


@traceable(name="internal_qa_pipeline")
def traced_internal_run(initial_state: dict) -> dict:
    return text_pipeline.graph.invoke(initial_state)
