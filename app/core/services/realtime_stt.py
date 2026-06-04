"""
OpenAI Realtime API GA 기반 STT.

stream_events() : WebSocket 스트리밍 → interim delta + final 이벤트 방출.
transcribe()    : gpt-4o-transcribe HTTP API 로 단일 결과 반환.

GA API 핵심 스펙:
  - 연결 모델  : gpt-realtime  (speech-to-speech 계열, REALTIME_STT_MODEL 로 오버라이드 가능)
  - 전사 모델  : gpt-realtime-whisper  (세션 내 transcription.model 필드)
  - 세션 타입  : "transcription"  (STT 전용 — LLM 응답 생성 없음)
  - 오디오 포맷: pcm16, 24 kHz mono  (클라이언트 16 kHz → 내부 24 kHz 업샘플링)
  - turn_detection: None  (수동 buffer commit 방식)
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import wave
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

import numpy as np
import openai

logger = logging.getLogger(__name__)

_TARGET_HZ = 24000  # gpt-realtime-whisper 요구 샘플레이트


# ──────────────────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────────────────

@dataclass
class STTStreamEvent:
    transcript: str
    language_code: str
    is_final: bool
    confidence: float = 0.0


@dataclass
class STTResult:
    transcript: str
    language_code: str
    confidence: float = 0.0


@dataclass
class RealtimeSTTConfig:
    # WebSocket 연결 모델 — gpt-realtime-whisper 는 transcription.model 에만 사용 가능
    model: str = field(default_factory=lambda: os.environ.get("REALTIME_STT_MODEL", "gpt-realtime"))
    # 세션 내 전사 모델 (gpt-realtime-whisper: 최저 레이턴시 STT 전용)
    transcription_model: str = "gpt-realtime-whisper"
    # None → 자동 감지
    language: Optional[str] = None
    # 클라이언트 전송 샘플레이트 (내부에서 24 kHz 로 업샘플링)
    input_sample_rate_hz: int = 16000
    audio_channel_count: int = 1


# ──────────────────────────────────────────────────────────
# OpenAIRealtimeSTT
# ──────────────────────────────────────────────────────────

class OpenAIRealtimeSTT:
    _LANG_MAP: dict = {
        "ko": "ko", "ko-KR": "ko", "ko-kr": "ko",
        "en": "en", "en-US": "en", "en-us": "en", "en-GB": "en", "en-gb": "en",
        "zh": "zh", "zh-CN": "zh", "zh-cn": "zh", "zh-TW": "zh", "zh-tw": "zh",
        "ja": "ja", "ja-JP": "ja", "ja-jp": "ja",
        "fr": "fr", "fr-FR": "fr",
        "es": "es", "es-ES": "es",
        "de": "de", "de-DE": "de",
    }

    _SPOKEN_LANG_NAMES: dict = {
        "korean": "ko", "english": "en", "japanese": "ja",
        "chinese": "zh", "french": "fr", "spanish": "es",
        "german": "de", "russian": "ru", "arabic": "ar",
        "portuguese": "pt", "italian": "it", "dutch": "nl",
        "thai": "th", "vietnamese": "vi", "indonesian": "id",
    }

    _SHORT_TO_ISO639: dict = {
        "ko": "ko", "en": "en", "zh": "zh", "ja": "ja",
        "fr": "fr", "es": "es", "de": "de",
    }

    def __init__(self, config: Optional[RealtimeSTTConfig] = None):
        self.config = config or RealtimeSTTConfig()
        self._sync_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ── 언어 유틸 ─────────────────────────────────────────

    def _normalize_lang(self, raw: Optional[str]) -> str:
        if not raw:
            return "ko"
        mapped = (
            self._LANG_MAP.get(raw)
            or self._LANG_MAP.get(raw.lower())
            or self._SPOKEN_LANG_NAMES.get(raw.lower())
        )
        if mapped:
            return mapped
        return raw.split("-")[0].lower()

    def _to_iso639(self, code: Optional[str]) -> Optional[str]:
        """None → None (자동 감지), 그 외 → ISO-639-1 2자리 코드."""
        if not code:
            return None
        normalized = self._normalize_lang(code)
        return self._SHORT_TO_ISO639.get(normalized, normalized)

    def _detect_lang_from_text(self, text: str) -> str:
        """텍스트 문자 분포로 언어 추정 (API 가 언어를 반환하지 않을 때 fallback)."""
        if not text:
            return "ko"
        hangul = sum(1 for c in text if "가" <= c <= "힣")
        kana   = sum(1 for c in text if "ぁ" <= c <= "ヿ")
        cjk    = sum(1 for c in text if "一" <= c <= "鿿")
        latin  = sum(1 for c in text if c.isalpha() and ord(c) < 128)

        if kana > 0:
            return "ja"

        scores = {"ko": hangul, "zh": cjk, "en": latin}
        dominant, dominant_count = max(scores.items(), key=lambda x: x[1])
        total_script = max(sum(scores.values()), 1)
        if dominant_count / total_script >= 0.4:
            return dominant
        return "en" if latin > 0 else "ko"

    # ── 오디오 유틸 ───────────────────────────────────────

    def _pcm_to_wav(self, pcm_bytes: bytes, *, sample_rate_hz: Optional[int] = None) -> bytes:
        """transcribe() 단건 HTTP 호출용 WAV 변환."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.audio_channel_count)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate_hz or self.config.input_sample_rate_hz)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def _resample_to_24k(self, pcm_bytes: bytes, src_hz: int) -> bytes:
        """16-bit mono PCM을 24 kHz로 업샘플링.

        scipy 우선, 없으면 numpy 선형 보간 사용.
        gpt-realtime-whisper는 24 kHz PCM 만 허용하므로 반드시 변환 필요.
        """
        if src_hz == _TARGET_HZ:
            return pcm_bytes

        try:
            from scipy.signal import resample_poly
            from math import gcd
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            g = gcd(_TARGET_HZ, src_hz)
            resampled = resample_poly(samples, _TARGET_HZ // g, src_hz // g)
        except ImportError:
            # scipy 없음 → numpy 선형 보간
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            n_out = int(len(samples) * _TARGET_HZ / src_hz)
            x_old = np.linspace(0, 1, len(samples))
            x_new = np.linspace(0, 1, n_out)
            resampled = np.interp(x_new, x_old, samples)

        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    # ── transcribe (동기, HTTP API 단건 전사) ─────────────

    def transcribe(
        self,
        audio_bytes: bytes,
        primary_language: Optional[str] = None,
        speech_phrases: Optional[List[str]] = None,
    ) -> STTResult:
        """단건 전사. LangGraph stt_node 에서 호출된다."""
        if not audio_bytes:
            return STTResult(transcript="", language_code="ko", confidence=0.0)

        wav_bytes = self._pcm_to_wav(audio_bytes)
        lang = self._to_iso639(primary_language or self.config.language)

        resp = self._sync_client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
            language=lang,
            response_format="verbose_json",
        )
        transcript = (resp.text or "").strip()
        raw_lang = getattr(resp, "language", None)
        detected = self._normalize_lang(raw_lang) if raw_lang else self._detect_lang_from_text(transcript)

        return STTResult(transcript=transcript, language_code=detected, confidence=0.0)

    # ── stream_events (비동기, Realtime WebSocket) ────────

    async def stream_events(
        self,
        audio_q: "asyncio.Queue[Optional[bytes]]",
        *,
        primary_language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        enable_punctuation: Optional[bool] = None,
        interim_results: bool = True,
        single_utterance: bool = False,
        speech_phrases: Optional[List[str]] = None,
    ) -> AsyncIterator[STTStreamEvent]:
        """Realtime WebSocket 스트리밍 STT.

        GA API 스펙:
          - 연결 모델: gpt-realtime  (type="transcription" 세션 지원)
          - 전사 모델: gpt-realtime-whisper  (session.audio.input.transcription.model)
          - 세션 타입: "transcription"  → LLM 응답 없이 전사만 수행
          - turn_detection: None  → 수동 buffer.commit() 으로 전사 트리거
          - 오디오: pcm16 24 kHz  → 클라이언트 16 kHz 는 내부 업샘플링
        """
        del enable_punctuation, single_utterance, speech_phrases

        from openai import AsyncOpenAI
        async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        lang = self._to_iso639(primary_language or self.config.language)
        src_hz = sample_rate_hz or self.config.input_sample_rate_hz

        transcription_cfg: dict = {"model": self.config.transcription_model}
        if lang:
            transcription_cfg["language"] = lang

        logger.warning(
            "realtime_stt | model=%s | transcription_model=%s | language=%s",
            self.config.model, self.config.transcription_model, lang,
        )

        async with async_client.realtime.connect(model=self.config.model) as conn:
            # transcription 전용 세션: type="transcription" 필수 (OpenAI Realtime Transcription API)
            # - output_modalities=["text"]: 오디오 출력 비활성화
            # - audio.input.turn_detection: None → 수동 commit 으로 전사 트리거
            # - response.create 를 보내지 않으므로 LLM 응답 생성 없음
            await conn.session.update(session={
                "type": "transcription",
                "output_modalities": ["text"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": transcription_cfg,
                        "turn_detection": None,
                    }
                },
            })

            cumulative = ""

            send_error: Optional[Exception] = None

            async def _send_audio():
                nonlocal send_error
                try:
                    while True:
                        chunk = await audio_q.get()
                        if chunk is None:
                            break
                        if chunk:
                            pcm = self._resample_to_24k(chunk, src_hz)
                            await conn.input_audio_buffer.append(
                                audio=base64.b64encode(pcm).decode()
                            )
                    await conn.input_audio_buffer.commit()
                except Exception as exc:
                    logger.warning("realtime_stt send error: %s", exc)
                    send_error = exc
                    await conn.close()

            send_task = asyncio.create_task(_send_audio())

            try:
                async for event in conn:
                    t = event.type

                    if t == "conversation.item.input_audio_transcription.delta":
                        if interim_results:
                            delta = getattr(event, "delta", "") or ""
                            if delta:
                                cumulative += delta
                                yield STTStreamEvent(
                                    transcript=cumulative,
                                    language_code=lang or "ko",
                                    is_final=False,
                                    confidence=0.0,
                                )

                    elif t == "conversation.item.input_audio_transcription.completed":
                        transcript = (getattr(event, "transcript", "") or "").strip()
                        logger.warning(
                            "realtime_stt | completed | language=%s | transcript=%s",
                            lang, transcript,
                        )
                        if transcript:
                            detected = self._detect_lang_from_text(transcript) if transcript else (lang or "ko")
                            yield STTStreamEvent(
                                transcript=transcript,
                                language_code=detected,
                                is_final=True,
                                confidence=0.0,
                            )
                        break

                    elif t == "error":
                        err = getattr(event, "error", str(event))
                        raise RuntimeError(f"Realtime API error: {err}")

                if send_error is not None:
                    raise send_error

            finally:
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
