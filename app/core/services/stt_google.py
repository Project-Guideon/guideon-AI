from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from google.cloud.speech_v1 import SpeechClient
from google.cloud.speech_v1 import types as speech_types


@dataclass
class STTStreamEvent:
    transcript: str
    language_code: str          # "ko"|"en"|...
    is_final: bool
    confidence: float = 0.0


@dataclass
class STTConfig:
    primary_language: str = "ko-KR"
    alternative_languages: List[str] = field(default_factory=lambda: ["en-US", "zh-CN", "ja-JP"])
    sample_rate_hz: int = 16000
    enable_punctuation: bool = True
    encoding: speech_types.RecognitionConfig.AudioEncoding = (
        speech_types.RecognitionConfig.AudioEncoding.LINEAR16
    )


@dataclass
class STTResult:
    transcript: str
    language_code: str   # 2자리: "ko" | "en" | "zh" | "ja" 등
    confidence: float = 0.0


class GoogleSTT:
    _LANG_MAP: dict = {
        "ko-KR": "ko", "ko-kr": "ko", "ko": "ko",
        "en-US": "en", "en-us": "en", "en-GB": "en", "en-gb": "en", "en": "en",
        "zh-CN": "zh", "zh-cn": "zh", "zh-TW": "zh", "zh-tw": "zh", "zh": "zh",
        "ja-JP": "ja", "ja-jp": "ja", "ja": "ja",
        "fr-FR": "fr", "fr": "fr",
        "es-ES": "es", "es": "es",
    }

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self.client = SpeechClient()  # v1 고정

    def _normalize_lang(self, raw: str) -> str:
        return self._LANG_MAP.get(raw) or raw.split("-")[0].lower()

    def transcribe(self, audio_bytes: bytes) -> STTResult:
        if not audio_bytes:
            return STTResult(transcript="", language_code="ko", confidence=0.0)

        audio = speech_types.RecognitionAudio(content=audio_bytes)
        cfg = speech_types.RecognitionConfig(
            encoding=self.config.encoding,
            sample_rate_hertz=self.config.sample_rate_hz,
            language_code=self.config.primary_language,
            alternative_language_codes=self.config.alternative_languages,
            enable_automatic_punctuation=self.config.enable_punctuation,
        )

        resp = self.client.recognize(config=cfg, audio=audio)

        if not resp.results:
            return STTResult(transcript="", language_code="ko", confidence=0.0)

        result = resp.results[0]
        alt = result.alternatives[0]
        transcript = (alt.transcript or "").strip()
        confidence = float(getattr(alt, "confidence", 0.0) or 0.0)

        raw_lang = getattr(result, "language_code", None) or self.config.primary_language
        language_code = self._normalize_lang(raw_lang)

        return STTResult(transcript=transcript, language_code=language_code, confidence=confidence)

    async def stream_events(
        self,
        audio_q: "asyncio.Queue[Optional[bytes]]",
        *,
        primary_language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        enable_punctuation: Optional[bool] = None,
        interim_results: bool = True,
        single_utterance: bool = False,
    ) -> AsyncIterator[STTStreamEvent]:
        """
        audio_q로부터 bytes 청크를 받아 Google Streaming STT에 흘려보내고,
        (interim/final) 결과 이벤트를 async generator로 방출한다.

        audio_q에 None을 넣으면 스트리밍 종료.
        """
        primary_language = primary_language or self.config.primary_language
        sample_rate_hz = sample_rate_hz or self.config.sample_rate_hz
        enable_punctuation = (
            self.config.enable_punctuation if enable_punctuation is None else enable_punctuation
        )

        recog_cfg = speech_types.RecognitionConfig(
            encoding=self.config.encoding,
            sample_rate_hertz=sample_rate_hz,
            language_code=primary_language,
            alternative_language_codes=self.config.alternative_languages,
            enable_automatic_punctuation=enable_punctuation,
        )
        streaming_cfg = speech_types.StreamingRecognitionConfig(
            config=recog_cfg,
            interim_results=interim_results,
            single_utterance=single_utterance,
        )

        loop = asyncio.get_running_loop()
        out_q: "asyncio.Queue[Optional[STTStreamEvent]]" = asyncio.Queue()

        def _run_streaming_recognize():
            """동기 스레드에서 Google streaming_recognize 돌리고 out_q로 결과를 넘김."""

            # ✅ 신버전용: 첫 request에 streaming_config 포함
            def req_gen_new():
                yield speech_types.StreamingRecognizeRequest(streaming_config=streaming_cfg)
                while True:
                    chunk = asyncio.run_coroutine_threadsafe(audio_q.get(), loop).result()
                    if chunk is None:
                        break
                    yield speech_types.StreamingRecognizeRequest(audio_content=chunk)

            # ✅ 구버전용: config는 함수 인자로 주고, request에는 audio_content만 넣음
            def req_gen_old():
                while True:
                    chunk = asyncio.run_coroutine_threadsafe(audio_q.get(), loop).result()
                    if chunk is None:
                        break
                    yield speech_types.StreamingRecognizeRequest(audio_content=chunk)

            try:
                # 1) 신버전 시도
                responses = self.client.streaming_recognize(requests=req_gen_new())
            except TypeError:
                # 2) 구버전 폴백 (네 에러가 딱 이 케이스)
                responses = self.client.streaming_recognize(streaming_cfg, req_gen_old())

            try:
                for resp in responses:
                    for result in resp.results:
                        if not result.alternatives:
                            continue
                        alt = result.alternatives[0]
                        transcript = (alt.transcript or "").strip()
                        conf = float(getattr(alt, "confidence", 0.0) or 0.0)

                        raw_lang = getattr(result, "language_code", None) or primary_language
                        lang2 = self._normalize_lang(raw_lang)

                        event = STTStreamEvent(
                            transcript=transcript,
                            language_code=lang2,
                            is_final=bool(result.is_final),
                            confidence=conf,
                        )
                        asyncio.run_coroutine_threadsafe(out_q.put(event), loop)
            finally:
                asyncio.run_coroutine_threadsafe(out_q.put(None), loop)

        worker_task = asyncio.create_task(asyncio.to_thread(_run_streaming_recognize))
        try:
            while True:
                ev = await out_q.get()
                if ev is None:
                    break
                yield ev
        finally:
            await worker_task