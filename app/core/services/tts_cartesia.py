"""
Cartesia TTS 서비스

Cartesia API(sonic-2 모델)를 사용하여 텍스트를 PCM 오디오로 변환합니다.
WebSocket 스트리밍을 통해 낮은 지연시간으로 오디오 청크를 수신합니다.
출력 포맷: pcm_s16le (16비트 부호형 정수 리틀엔디안, 24000Hz)

최적화: WebSocket 연결을 유지 재사용합니다.
  - 문장마다 새 연결을 열지 않아 연결 설정 지연(50~200ms)을 제거합니다.
  - 연결 오류 시 1회 자동 재연결합니다.
"""
from __future__ import annotations
import asyncio
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

    WebSocket 연결을 내부적으로 유지하며 재사용합니다.
    연결이 끊기면 자동으로 재연결합니다.

    사용 예시:
        tts = CartesiaTTS(api_key="sk_car_...")
        audio_bytes = await tts.synthesize_async("안녕하세요", "ko-KR", voice_id="...")
    """

    def __init__(
        self,
        api_key: str,
        voice_id: Optional[str] = None,
        model_id: str = "sonic-2",
        sample_rate: int = 24000,
    ):
        """
        Args:
            api_key: Cartesia API 키 (환경변수 CARTESIA_API_KEY)
            voice_id: 기본 음성 ID. None이면 마스코트별 voice_id를 필수로 지정해야 함
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
        # 재사용 WebSocket 연결 — None이면 _get_ws()에서 새로 생성
        self._ws = None
        # 연결 생성/재연결 시 중복 생성을 막기 위한 락
        self._ws_lock = asyncio.Lock()

    @staticmethod
    def map_language(language_code: str) -> str:
        """BCP-47 / ISO 639-1 언어 코드를 Cartesia가 수용하는 ISO 639-1 코드로 변환."""
        lang2 = (language_code or "ko").split("-")[0].lower()
        return _LANG_MAP.get(language_code) or _LANG_MAP.get(lang2, "ko")

    def get_audio_format(self) -> str:
        """Unity 클라이언트에 전달할 audio_format 필드값 반환."""
        return "pcm_s16le"

    async def _get_ws(self):
        """
        WebSocket 연결을 반환합니다. 없으면 새로 생성합니다.

        double-checked locking: 이미 연결이 있으면 락 없이 즉시 반환합니다.
        """
        if self._ws is not None:
            return self._ws
        async with self._ws_lock:
            # 락 획득 사이에 다른 코루틴이 생성했을 수 있으므로 재확인
            if self._ws is None:
                self._ws = await self._client.tts.websocket()
                logger.info("Cartesia WebSocket 연결 생성")
        return self._ws

    async def _reset_ws(self):
        """연결을 폐기합니다. 오류 발생 시 호출합니다."""
        async with self._ws_lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
            logger.info("Cartesia WebSocket 연결 폐기")

    async def synthesize_async(
        self,
        text: str,
        language_code: str = "ko-KR",
        voice_id: Optional[str] = None,
    ) -> bytes:
        """
        텍스트를 PCM 오디오 바이트로 변환합니다 (비동기).

        WebSocket 연결을 재사용하여 연결 설정 지연을 없앱니다.
        연결 오류 시 자동으로 재연결 후 1회 재시도합니다.

        Args:
            text: 합성할 텍스트
            language_code: 언어 코드 (예: "ko-KR", "ja-JP", "en-US")
            voice_id: 사용할 음성 ID. None이면 self.voice_id 사용

        Returns:
            PCM 오디오 바이트 (pcm_s16le, 24000Hz). 입력이 비어있으면 b"" 반환.
        """
        if not text or not text.strip():
            return b""

        lang = self.map_language(language_code)
        vid = voice_id or self.voice_id
        if not vid:
            raise ValueError("voice_id가 지정되지 않았습니다. 마스코트 ttsVoiceId를 설정하세요.")

        # 연결 오류 시 재연결 후 1회 재시도
        for attempt in range(2):
            try:
                ws = await self._get_ws()
                buf = bytearray()
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
                return bytes(buf)

            except Exception as exc:
                await self._reset_ws()
                if attempt == 1:
                    raise
                logger.warning("Cartesia WebSocket 오류, 재연결 후 재시도: %s", exc)

    async def clone_voice_async(
        self,
        filepath: str,
        name: str,
        language: str = "ko",
        mode: str = "similarity",
    ) -> tuple[str, str]:
        """
        음성 샘플 파일로 Cartesia 커스텀 보이스를 클로닝합니다.

        Args:
            filepath: 임시 저장된 오디오 파일 경로
            name: 생성할 보이스 이름
            language: 언어 코드 ISO 639-1 (기본값: "ko")
            mode: 클로닝 모드.
                  "clip" → embedding 리스트만 반환(voice_id 없음, 사용 불가)
                  "similarity" / "stability" → 이름 있는 보이스 생성, voice_id 반환

        Returns:
            (voice_id, name) 튜플 — voice_id를 tb_mascot.tts_voice_id에 저장
        """
        # voices.clone()은 동기 함수(httpx.post 사용) — 이벤트 루프 블로킹 방지
        result = await asyncio.to_thread(
            self._client.voices.clone,
            filepath=filepath,
            name=name,
            language=language,
            mode=mode,
        )
        # clip 이외 mode는 dict 반환
        if isinstance(result, dict):
            voice_id = result.get("id")
            voice_name = result.get("name", name)
        else:
            raise ValueError(f"예상치 못한 clone 반환값 타입: {type(result)}. mode를 확인하세요.")

        if not voice_id:
            raise ValueError(f"Cartesia 응답에 id 없음: {result}")

        return voice_id, voice_name
