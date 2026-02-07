# services/stt_google.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from google.cloud import speech


@dataclass
class STTConfig:
    language_code: str = "ko-KR"
    sample_rate_hz: int = 16000
    enable_punctuation: bool = True
    encoding: speech.RecognitionConfig.AudioEncoding = speech.RecognitionConfig.AudioEncoding.LINEAR16


class GoogleSTT:
    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self.client = speech.SpeechClient()

    def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""

        audio = speech.RecognitionAudio(content=audio_bytes)
        cfg = speech.RecognitionConfig(
            encoding=self.config.encoding,
            sample_rate_hertz=self.config.sample_rate_hz,
            language_code=self.config.language_code,
            enable_automatic_punctuation=self.config.enable_punctuation,
        )
        resp = self.client.recognize(config=cfg, audio=audio)

        if not resp.results:
            return ""

        # 여러 result 합치고 싶으면 join 하면 됨
        return resp.results[0].alternatives[0].transcript.strip()
