# services/llm_openai.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from app.core.services.rag_pgvector import RetrievedChunk


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 400


class OpenAILLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

    def generate(self, query: str, contexts: List[RetrievedChunk]) -> str:
        ctx = "\n\n".join(
            [f"[{i+1}] {c.content}" for i, c in enumerate(contexts)]
        )

        system = (
            "너는 한국어로 답하는 관광/가이드 도우미야.\n"
            "아래 CONTEXT에 있는 정보만 근거로 답해.\n"
            "CONTEXT에 없으면 '자료에 없음'이라고 말해.\n"
            "답변은 2~4문장으로 짧고 자연스럽게.\n"
        )
        user = f"CONTEXT:\n{ctx}\n\nQUESTION:\n{query}"

        resp = self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return resp.content.strip()
