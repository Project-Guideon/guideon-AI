from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, AsyncIterator, AsyncIterable
from google.cloud import texttospeech_v1 as texttospeech
from google.api_core.client_options import ClientOptions


@dataclass
class TTSConfig:
    language_code: str = "ko-KR"
    voice_name: Optional[str] = None
    speaking_rate: float = 1.0
    pitch: float = 0.0
    audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.MP3


class GoogleTTS:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.client = texttospeech.TextToSpeechClient()

    def get_audio_format(self) -> str:
        if self.config.audio_encoding == texttospeech.AudioEncoding.MP3:
            return "mp3"
        if self.config.audio_encoding == texttospeech.AudioEncoding.LINEAR16:
            return "wav"
        return "bin"

    def synthesize(
        self,
        text: str,
        language_code: Optional[str] = None,
        voice_name: Optional[str] = None,
    ) -> bytes:
        if not text or not text.strip():
            return b""

        lang = language_code or self.config.language_code
        vname = voice_name or self.config.voice_name

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_kwargs = {"language_code": lang}
        if vname:
            voice_kwargs["name"] = vname

        voice = texttospeech.VoiceSelectionParams(**voice_kwargs)

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

# 실시간 tts
@dataclass
class StreamingTTSConfig:
    language_code: str = "ko-KR"
    voice_name: Optional[str] = None
    # streaming에서는 MP3 말고 PCM/OGG_OPUS 권장
    # Google streaming TTS는 Chirp 3: HD voices 기준으로 ALAW/MULAW/OGG_OPUS/PCM 지원
    audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.PCM
    # 지역 엔드포인트를 쓰고 싶으면 예: "asia-northeast1-texttospeech.googleapis.com"
    endpoint: Optional[str] = None


class StreamingGoogleTTS:
    def __init__(self, config: Optional[StreamingTTSConfig] = None):
        self.config = config or StreamingTTSConfig()
        if self.config.endpoint:
            self.client = texttospeech.TextToSpeechAsyncClient(
                client_options=ClientOptions(api_endpoint=self.config.endpoint)
            )
        else:
            self.client = texttospeech.TextToSpeechAsyncClient()

    def get_audio_format(self) -> str:
        if self.config.audio_encoding == texttospeech.AudioEncoding.OGG_OPUS:
            return "ogg_opus"
        if self.config.audio_encoding == texttospeech.AudioEncoding.PCM:
            return "pcm"
        if self.config.audio_encoding == texttospeech.AudioEncoding.ALAW:
            return "alaw"
        if self.config.audio_encoding == texttospeech.AudioEncoding.MULAW:
            return "mulaw"
        return "bin"

    async def stream_synthesize(
        self,
        text_chunks: AsyncIterable[str],
        language_code: Optional[str] = None,
        voice_name: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        lang = language_code or self.config.language_code
        vname = voice_name or self.config.voice_name

        voice_kwargs = {"language_code": lang}
        if vname:
            voice_kwargs["name"] = vname

        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(**voice_kwargs),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=self.config.audio_encoding,
            ),
        )

        async def request_generator():
            # 첫 요청은 반드시 config
            yield texttospeech.StreamingSynthesizeRequest(
                streaming_config=streaming_config
            )

            async for chunk in text_chunks:
                if not chunk or not chunk.strip():
                    continue
                yield texttospeech.StreamingSynthesizeRequest(
                    input=texttospeech.StreamingSynthesisInput(text=chunk)
                )

        stream = await self.client.streaming_synthesize(
            requests=request_generator()
        )

        async for response in stream:
            if response.audio_content:
                yield response.audio_content