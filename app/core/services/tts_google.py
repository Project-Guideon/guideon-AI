# services/tts_google.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from google.cloud import texttospeech


@dataclass
class TTSConfig:
    language_code: str = "ko-KR"
    # voice_name 예: "ko-KR-Standard-A" (프로젝트에서 원하는 걸로 고정 가능)
    voice_name: Optional[str] = None
    speaking_rate: float = 1.0
    pitch: float = 0.0
    audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.MP3


class GoogleTTS:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.client = texttospeech.TextToSpeechClient()

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            text = "음성으로 변환할 내용이 없습니다."

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=self.config.language_code,
            name=self.config.voice_name if self.config.voice_name else None,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=self.config.audio_encoding,
            speaking_rate=self.config.speaking_rate,
            pitch=self.config.pitch,
        )

        resp = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return resp.audio_content
