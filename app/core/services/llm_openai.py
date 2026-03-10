# services/llm_openai.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, AsyncIterator
import re

from openai import OpenAI
from app.core.services.rag_pgvector import RetrievedChunk
from langsmith.wrappers import wrap_openai


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 150


_SENTENCE_RE = re.compile(
    r'(.+?[.!?]|.+?다\.|.+?요\.|.+?니다\.)',
    re.DOTALL
)


class OpenAILLM:
    def __init__(self, cfg: LLMConfig = None):
        self.cfg = cfg or LLMConfig()
        base_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0)
        self.client = wrap_openai(base_client)

    # ---------------------------------
    # 기본 chat 호출
    # ---------------------------------
    def chat(self, messages: list, max_tokens: int = None) -> str:
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    # ---------------------------------
    # 기존 방식 (동기)
    # ---------------------------------
    def generate(self, query: str, contexts: List[RetrievedChunk]) -> str:

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

        return (response.choices[0].message.content or "").strip()

    # ---------------------------------
    # LLM 토큰 스트리밍
    # ---------------------------------
    async def stream_generate(
        self,
        query: str,
        contexts: List[RetrievedChunk],
    ) -> AsyncIterator[str]:

        if not contexts:
            yield "관련 정보를 찾을 수 없습니다."
            return

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

        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            stream=True,
        )

        for event in stream:
            delta = event.choices[0].delta.content or ""
            if delta:
                yield delta

    # ---------------------------------
    # 문장 단위 스트리밍 (TTS용)
    # ---------------------------------
    async def stream_generate_sentences(
        self,
        query: str,
        contexts: List[RetrievedChunk],
    ) -> AsyncIterator[str]:

        buffer = ""

        async for delta in self.stream_generate(query, contexts):
            buffer += delta

            while True:
                m = _SENTENCE_RE.match(buffer)
                if not m:
                    break

                sentence = m.group(1).strip()
                buffer = buffer[m.end():].lstrip()

                if sentence:
                    yield sentence

        if buffer.strip():
            yield buffer.strip()