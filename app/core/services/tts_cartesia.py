"""
Cartesia TTS 서비스

Cartesia API(sonic-2 모델)를 사용하여 텍스트를 PCM 오디오로 변환합니다.
WebSocket 스트리밍을 통해 낮은 지연시간으로 오디오 청크를 수신합니다.
출력 포맷: pcm_s16le (16비트 부호형 정수 리틀엔디안, 24000Hz)
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# TTS 언어 코드 → Cartesia 언어 코드 매핑
# Google STT는 BCP-47 형식(ko-KR)을 사용하고, Cartesia는 ISO 639-1(ko)을 사용
_LANG_MAP: dict[str, str] = {
    "ko": "ko", "ko-KR": "ko",
    "en": "en", "en-US": "en",
    "ja": "ja", "ja-JP": "ja",
    "zh": "zh", "cmn-CN": "zh", "cmn-Hans-CN": "zh",
}


class CartesiaTTS:
    """
    Cartesia TTS 클라이언트.

    사용 예시:
        tts = CartesiaTTS(api_key="sk_car_...", voice_id="voice-id-here")
        audio_bytes = await tts.synthesize_async("안녕하세요", "ko-KR")
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "sonic-2",
        sample_rate: int = 24000,
    ):
        """
        Args:
            api_key: Cartesia API 키 (환경변수 CARTESIA_API_KEY)
            voice_id: 사용할 음성 ID (환경변수 CARTESIA_VOICE_ID)
            model_id: Cartesia 모델 ID (기본값: sonic-2)
            sample_rate: 출력 샘플레이트(Hz) (기본값: 24000)
        """
        from cartesia import AsyncCartesia  # 런타임에 import → 패키지 없어도 모듈 로드 가능
        self._client = AsyncCartesia(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id
        # Cartesia 출력 포맷: 컨테이너 없는 raw PCM
        self._output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
        }

    @staticmethod
    def map_language(language_code: str) -> str:
        """BCP-47 / ISO 639-1 언어 코드를 Cartesia가 수용하는 ISO 639-1 코드로 변환."""
        lang2 = (language_code or "ko").split("-")[0].lower()
        return _LANG_MAP.get(language_code) or _LANG_MAP.get(lang2, "ko")

    def get_audio_format(self) -> str:
        """Unity 클라이언트에 전달할 audio_format 필드값 반환."""
        return "pcm_s16le"

    async def synthesize_async(
        self,
        text: str,
        language_code: str = "ko-KR",
        voice_id: Optional[str] = None,
    ) -> bytes:
        """
        텍스트를 PCM 오디오 바이트로 변환합니다 (비동기).

        WebSocket 스트리밍으로 모든 청크를 수집한 후 단일 bytes로 반환합니다.

        Args:
            text: 합성할 텍스트
            language_code: 언어 코드 (예: "ko-KR", "ja-JP", "en-US")
            voice_id: 사용할 음성 ID. None이면 self.voice_id 사용 (Phase 2: 마스코트별 음성 지원)

        Returns:
            PCM 오디오 바이트 (pcm_s16le, 24000Hz). 입력이 비어있으면 b"" 반환.
        """
        if not text or not text.strip():
            return b""

        lang = self.map_language(language_code)
        vid = voice_id or self.voice_id
        buf = bytearray()

        # websocket()은 async 메서드 — await으로 연결 후 finally에서 반드시 close
        ws = await self._client.tts.websocket()
        try:
            # send()는 stream=True 시 AsyncGenerator 반환 — await 후 async for로 순회
            audio_gen = await ws.send(
                model_id=self.model_id,
                transcript=text,
                voice_id=vid,           # 문자열 직접 전달 (dict 아님)
                output_format=self._output_format,
                language=lang,
                stream=True,
            )
            async for chunk in audio_gen:
                if chunk.get("audio"):  # chunk는 TypedDict — .audio 아닌 ["audio"]
                    buf.extend(chunk["audio"])
        finally:
            await ws.close()

        return bytes(buf)
