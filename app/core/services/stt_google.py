from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from google.cloud import speech


@dataclass
class STTConfig:
    primary_language: str = "ko-KR"
    # 다국어 감지용 후보 언어 목록 (설정해야 결과에 language_code 필드가 채워짐)
    alternative_languages: List[str] = field(
        default_factory=lambda: ["en-US", "zh-CN", "ja-JP"]
    )
    sample_rate_hz: int = 16000  # 유니티에서 16000Hz로 맞춰야 함
    enable_punctuation: bool = True
    encoding: speech.RecognitionConfig.AudioEncoding = (
        speech.RecognitionConfig.AudioEncoding.LINEAR16
    )


@dataclass
class STTResult:
    transcript: str
    language_code: str   # 2자리: "ko" | "en" | "zh" | "ja" 등
    confidence: float = 0.0


class GoogleSTT:
    # Google STT 가 반환하는 BCP-47 태그 → 2자리 코드 매핑
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
        self.client = speech.SpeechClient()
        # GOOGLE_APPLICATION_CREDENTIALS 환경변수로 인증키 경로 설정 필요

    def _normalize_lang(self, raw: str) -> str:
        """BCP-47 → 2자리 코드. 매핑에 없으면 앞 2자리 소문자로 fallback."""
        return self._LANG_MAP.get(raw) or raw.split("-")[0].lower()

    def transcribe(self, audio_bytes: bytes) -> STTResult:
        if not audio_bytes:
            return STTResult(transcript="", language_code="ko", confidence=0.0)

        audio = speech.RecognitionAudio(content=audio_bytes)
        cfg = speech.RecognitionConfig(
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
        transcript = result.alternatives[0].transcript.strip()
        confidence = float(result.alternatives[0].confidence or 0.0)

        # alternative_language_codes 설정 시 result.language_code 에 감지 언어가 들어옴
        raw_lang = (
            getattr(result, "language_code", None)
            or self.config.primary_language
        )
        language_code = self._normalize_lang(raw_lang)

        return STTResult(
            transcript=transcript,
            language_code=language_code,
            confidence=confidence,
        )
