# main_fastapi.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from dotenv import load_dotenv

from app.core.services.stt_google import GoogleSTT, STTConfig
from app.core.services.tts_google import GoogleTTS, TTSConfig
from app.core.services.rag_pgvector import PgVectorRAG
from app.core.services.llm_openai import OpenAILLM, LLMConfig
from app.core.services.pipeline import VoicePipeline


load_dotenv()

app = FastAPI(title="Guideon Voice QA")

stt = GoogleSTT(STTConfig(language_code="ko-KR", sample_rate_hz=16000))
tts = GoogleTTS(TTSConfig(language_code="ko-KR"))
rag = PgVectorRAG(model_name="paraphrase-multilingual-mpnet-base-v2")
llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", temperature=0.7, max_tokens=500))

pipeline = VoicePipeline(stt=stt, rag=rag, llm=llm, tts=tts)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"ok": True, "msg": "Go to /docs"}


@app.post("/voice_qa")
async def voice_qa(audio: UploadFile = File(...), site_id: int = 1, k: int = 5):
    audio_bytes = await audio.read()
    result = pipeline.run(audio_bytes, site_id=site_id, k=k)

    return Response(
        content=result.voice_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=answer.mp3"},
    )


@app.post("/text_qa") # 텍스트로 들어왔을때 택스트 답변 과정 사용할지 안할지는 미정
async def text_qa(query: str, site_id: int = 1, k: int = 5):
    contexts = rag.retrieve(query=query, site_id=site_id, k=k)
    answer = llm.generate(query=query, contexts=contexts)
    return JSONResponse({"query": query, "answer": answer})
