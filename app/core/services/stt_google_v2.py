"""
Google Cloud Speech-to-Text v2 스트리밍 클라이언트
- Chirp 3 (또는 GOOGLE_STT_MODEL 환경변수) 모델 사용
- language_codes 리스트로 한영 혼용(코드 스위칭) 지원
- 기존 stt_google.py(v1)와 동일한 인터페이스 유지 (drop-in replacement)
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


# ──────────────────────────────────────────────
# 공유 데이터 클래스 (v1과 동일 인터페이스)
# ──────────────────────────────────────────────

@dataclass
class STTStreamEvent:
    transcript: str
    language_code: str   # "ko" | "en" | ...
    is_final: bool
    confidence: float = 0.0


@dataclass
class STTResult:
    transcript: str
    language_code: str
    confidence: float = 0.0


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

@dataclass
class STTConfig:
    # v2는 language_codes 리스트로 코드 스위칭 지원
    language_codes: List[str] = field(default_factory=lambda: ["ko-KR", "en-US"])
    sample_rate_hz: int = 16000
    audio_channel_count: int = 1
    enable_punctuation: bool = True
    # 환경 변수에서 읽어온 값을 기본값으로 사용
    project_id: str = field(default_factory=lambda: os.environ.get("GOOGLE_PROJECT_ID", ""))
    location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_REGION", "us"))
    model: str = field(default_factory=lambda: os.environ.get("GOOGLE_STT_MODEL", "chirp_2"))
    # 사이트별 고유명사 힌트 (adaptation phrase sets 대신 adaptation으로 처리)
    default_speech_phrases: List[str] = field(default_factory=lambda: [
        "근정전", "사정전", "경복궁", "창덕궁", "창경궁", "덕수궁", "경회루",
        "향원정", "교태전", "강녕전", "집옥재", "자경전", "흥례문", "광화문",
    ])


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────

class GoogleSTTV2:
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
        # global이 아닌 리전(us, eu 등)은 리전 전용 엔드포인트로 연결해야 함
        if self.config.location != "global":
            opts = ClientOptions(
                api_endpoint=f"{self.config.location}-speech.googleapis.com"
            )
            self.client = SpeechClient(client_options=opts)
        else:
            self.client = SpeechClient()
        self._recognizer = (
            f"projects/{self.config.project_id}"
            f"/locations/{self.config.location}"
            f"/recognizers/_"
        )

    def _normalize_lang(self, raw: str) -> str:
        return self._LANG_MAP.get(raw) or self._LANG_MAP.get(raw.lower()) or raw.split("-")[0].lower()

    def _build_recognition_config(
        self,
        language_codes: Optional[List[str]] = None,
        sample_rate_hz: Optional[int] = None,
        enable_punctuation: Optional[bool] = None,
        speech_phrases: Optional[List[str]] = None,
    ) -> cloud_speech.RecognitionConfig:
        lang_codes = language_codes or self.config.language_codes
        rate = sample_rate_hz or self.config.sample_rate_hz
        punctuation = self.config.enable_punctuation if enable_punctuation is None else enable_punctuation

        cfg = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=rate,
                audio_channel_count=self.config.audio_channel_count,
            ),
            language_codes=lang_codes,
            model=self.config.model,
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=punctuation,
            ),
        )

        # Chirp 계열 모델은 SpeechAdaptation 미지원 — language_codes로 코드 스위칭 처리
        is_chirp = self.config.model.startswith("chirp")
        if not is_chirp:
            all_phrases = list(self.config.default_speech_phrases)
            if speech_phrases:
                all_phrases.extend(speech_phrases)
            if all_phrases:
                cfg.adaptation = cloud_speech.SpeechAdaptation(
                    phrase_sets=[
                        cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                            inline_phrase_set=cloud_speech.PhraseSet(
                                phrases=[
                                    cloud_speech.PhraseSet.Phrase(value=p, boost=10.0)
                                    for p in all_phrases
                                ]
                            )
                        )
                    ]
                )

        return cfg

    # ──────────────────────────────────────────
    # 배치 인식 (비스트리밍)
    # ──────────────────────────────────────────

    def transcribe(
        self,
        audio_bytes: bytes,
        speech_phrases: Optional[List[str]] = None,
    ) -> STTResult:
        if not audio_bytes:
            return STTResult(transcript="", language_code="ko", confidence=0.0)

        cfg = self._build_recognition_config(speech_phrases=speech_phrases)
        request = cloud_speech.RecognizeRequest(
            recognizer=self._recognizer,
            config=cfg,
            content=audio_bytes,
        )
        resp = self.client.recognize(request=request)

        if not resp.results:
            return STTResult(transcript="", language_code="ko", confidence=0.0)

        result = resp.results[0]
        alt = result.alternatives[0]
        transcript = (alt.transcript or "").strip()
        confidence = float(getattr(alt, "confidence", 0.0) or 0.0)
        raw_lang = getattr(result, "language_code", None) or self.config.language_codes[0]
        return STTResult(
            transcript=transcript,
            language_code=self._normalize_lang(raw_lang),
            confidence=confidence,
        )

    # ──────────────────────────────────────────
    # 스트리밍 인식
    # ──────────────────────────────────────────

    async def stream_events(
        self,
        audio_q: "asyncio.Queue[Optional[bytes]]",
        *,
        primary_language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        enable_punctuation: Optional[bool] = None,
        interim_results: bool = True,
        single_utterance: bool = False,  # v2 StreamingRecognitionFeatures 미지원 — 무시됨
        speech_phrases: Optional[List[str]] = None,
    ) -> AsyncIterator[STTStreamEvent]:
        """
        v1 GoogleSTT.stream_events()와 동일한 시그니처.
        primary_language가 주어지면 language_codes의 첫 번째 자리에 배치.
        """
        # primary_language가 들어오면 맨 앞에 두고 나머지는 config 기본값 유지
        if primary_language and primary_language not in self.config.language_codes:
            lang_codes = [primary_language] + [
                l for l in self.config.language_codes if l != primary_language
            ]
        else:
            lang_codes = list(self.config.language_codes)

        recog_cfg = self._build_recognition_config(
            language_codes=lang_codes,
            sample_rate_hz=sample_rate_hz,
            enable_punctuation=enable_punctuation,
            speech_phrases=speech_phrases,
        )

        streaming_cfg = cloud_speech.StreamingRecognitionConfig(
            config=recog_cfg,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=interim_results,
            ),
        )

        recognizer = self._recognizer
        loop = asyncio.get_running_loop()
        out_q: "asyncio.Queue[Optional[STTStreamEvent]]" = asyncio.Queue()

        def _run_streaming_recognize():
            def req_gen():
                # 첫 번째 요청: recognizer + streaming_config
                yield cloud_speech.StreamingRecognizeRequest(
                    recognizer=recognizer,
                    streaming_config=streaming_cfg,
                )
                # 이후 요청: 오디오 청크
                while True:
                    chunk = asyncio.run_coroutine_threadsafe(audio_q.get(), loop).result()
                    if chunk is None:
                        break
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

            try:
                responses = self.client.streaming_recognize(requests=req_gen())
                for resp in responses:
                    for result in resp.results:
                        if not result.alternatives:
                            continue
                        alt = result.alternatives[0]
                        transcript = (alt.transcript or "").strip()
                        conf = float(getattr(alt, "confidence", 0.0) or 0.0)
                        raw_lang = getattr(result, "language_code", None) or lang_codes[0]
                        event = STTStreamEvent(
                            transcript=transcript,
                            language_code=self._normalize_lang(raw_lang),
                            is_final=bool(result.is_final),
                            confidence=conf,
                        )
                        asyncio.run_coroutine_threadsafe(out_q.put(event), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(out_q.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(out_q.put(None), loop)

        worker_task = asyncio.create_task(asyncio.to_thread(_run_streaming_recognize))
        try:
            while True:
                ev = await out_q.get()
                if ev is None:
                    break
                if isinstance(ev, Exception):
                    raise ev
                yield ev
        finally:
            await worker_task
