# services/llm_openai.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List

from openai import OpenAI
from app.core.services.rag_pgvector import RetrievedChunk


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2 #낮을수록  RAG 답변이 더 사실적이고 일관되게 나옴 (0.0~1.0)
    max_tokens: int = 150  # TTS용: 짧고 명확하게 (150 ≈ 2~3문장)


class OpenAILLM:
    def __init__(self, cfg: LLMConfig = None):
        self.cfg = cfg or LLMConfig()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, query: str, contexts: List[RetrievedChunk]) -> str:
        #query는 stt로 만들어진 질문 텍스트, contexts는 rag로 검색된 관련 정보 리스트
        if not contexts:
            return "관련 정보를 찾을 수 없습니다."

        context = "\n\n".join(
            [f"[청크 {i+1}]\n{c.content}" for i, c in enumerate(contexts)]
        )

        system_prompt = (
            "당신은 관광 안내 음성 도우미입니다.\n"
            "제공된 정보(context)만 근거로, 2~5문장으로 짧고 자연스럽게 답하세요.\n"
            "불필요한 설명이나 나열은 하지 마세요.\n"
            "정보가 없으면 '관련 정보를 찾을 수 없습니다'라고만 답하세요."
        )

        user_message = (
            f"질문: {query}\n\n"
            f"제공 정보(Context):\n{context}\n\n"
            "위 정보를 기반으로 질문에 답변해주세요."
        )

        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        return response.choices[0].message.content.strip()
