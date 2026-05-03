from __future__ import annotations

import asyncio
import os
import queue as thread_queue
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


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
class STTConfig:
    # 여러 언어를 동시에 쓰려면 location이 us / eu / global 이어야 함
    language_codes: List[str] = field(
        default_factory=lambda: ["ko-KR", "en-US", "ja-JP"]
    )
    sample_rate_hz: int = 16000
    audio_channel_count: int = 1
    enable_punctuation: bool = True

    project_id: str = field(default_factory=lambda: os.environ.get("GOOGLE_PROJECT_ID", ""))
    location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_REGION", "us"))
    model: str = field(default_factory=lambda: os.environ.get("GOOGLE_STT_MODEL", "chirp_3"))

    default_speech_phrases: List[str] = field(default_factory=lambda: [
        "근정전", "사정전", "경복궁", "창덕궁", "창경궁", "덕수궁", "경회루",
        "향원정", "교태전", "강녕전", "집옥재", "자경전", "흥례문", "광화문",
    ])


class GoogleSTTV2:
    _LANG_MAP: dict = {
        "ko-KR": "ko", "ko-kr": "ko", "ko": "ko",
        "en-US": "en", "en-us": "en", "en-GB": "en", "en-gb": "en", "en": "en",
        "cmn-Hans-CN": "zh", "cmn-hans-cn": "zh",
        "zh-CN": "zh", "zh-cn": "zh", "zh-TW": "zh", "zh-tw": "zh", "zh": "zh",
        "ja-JP": "ja", "ja-jp": "ja", "ja": "ja",
    }

    # 기본 언어. 항상 STT 후보 1순위로 유지.
    BASE_LANGUAGE_CODE = "ko-KR"
    # STT가 감지할 수 있는 전체 후보 언어 목록
    CANDIDATE_LANGUAGE_CODES = ["ko-KR", "en-US", "ja-JP", "cmn-Hans-CN"]

    _MULTI_LANGUAGE_ALLOWED_LOCATIONS = {"global", "us", "eu"}
    
    # 단축 코드 → BCP-47 (API 요청용)
    _SHORT_TO_BCP47: dict = {
        "ko": "ko-KR",
        "en": "en-US",
        "zh": "cmn-Hans-CN",
        "ja": "ja-JP",
        "fr": "fr-FR",
        "es": "es-ES",
    }

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()

        if not self.config.project_id:
            raise ValueError("GOOGLE_PROJECT_ID 환경변수가 없습니다.")

        # language_codes가 여러 개인데 location이 허용되지 않으면 자동 보정
        if (
            len(self.config.language_codes) > 1
            and self.config.location not in self._MULTI_LANGUAGE_ALLOWED_LOCATIONS
        ):
            # 가장 안전하게 us로 강제
            self.config.location = "us"

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

    def _to_bcp47(self, code: str) -> str:
        """단축 코드(ko, zh 등)를 BCP-47 전체 코드로 변환."""
        return self._SHORT_TO_BCP47.get(code, code)

    def _resolve_language_codes(
        self,
        primary_language: Optional[str] = None,
    ) -> List[str]:
        """
        - 멀티언어 허용 리전(us/eu/global)이면 language_codes 여러 개 허용
        - 아니면 단일 언어만 사용
        """
        lang_codes = list(self.config.language_codes)

        if primary_language:
            primary_bcp47 = self._to_bcp47(primary_language)
            if primary_bcp47 in lang_codes:
                lang_codes.remove(primary_bcp47)
            lang_codes.insert(0, primary_bcp47)

        if self.config.location not in self._MULTI_LANGUAGE_ALLOWED_LOCATIONS:
            return [lang_codes[0]]

        return lang_codes

    # chirp_3가 지원하지 않는 언어 → chirp_2 폴백
    _CHIRP3_UNSUPPORTED = {"cmn-Hans-CN", "cmn-hans-cn", "zh-CN", "zh-TW"}

    def _resolve_model_and_langs(self, lang_codes: List[str]) -> tuple:
        """
        chirp_3 미지원 언어/제약 처리:
        - 주 언어가 미지원(zh 등) → long 모델 + 단일 언어
        - 주 언어가 지원되는 경우 → 후보 목록에서 미지원 언어 제거
          (Chinese/cmn-Hans-CN 은 chirp_3 다중 언어 모드 미지원)
        """
        if self.config.model != "chirp_3" or not lang_codes:
            return self.config.model, lang_codes

        if lang_codes[0] in self._CHIRP3_UNSUPPORTED:
            return "long", [lang_codes[0]]

        # primary 언어가 지원되는 경우, 후보에서 chirp_3 미지원 언어 제거
        filtered = [l for l in lang_codes if l not in self._CHIRP3_UNSUPPORTED]
        return self.config.model, filtered if filtered else [self.BASE_LANGUAGE_CODE]

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
        model, lang_codes = self._resolve_model_and_langs(lang_codes)

        cfg = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=rate,
                audio_channel_count=self.config.audio_channel_count,
            ),
            language_codes=lang_codes,
            model=model,
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=punctuation,
            ),
        )

        # Chirp 계열은 adaptation 미지원 → 비-Chirp 모델에서만 phrase boost
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

    def transcribe(
        self,
        audio_bytes: bytes,
        speech_phrases: Optional[List[str]] = None,
        primary_language: Optional[str] = None,
    ) -> STTResult:
        if not audio_bytes:
            return STTResult(transcript="", language_code="ko", confidence=0.0)

        lang_codes = self._resolve_language_codes(primary_language=primary_language)

        cfg = self._build_recognition_config(
            language_codes=lang_codes,
            speech_phrases=speech_phrases,
        )

        request = cloud_speech.RecognizeRequest(
            recognizer=self._recognizer,
            config=cfg,
            content=audio_bytes,
        )
        resp = self.client.recognize(request=request)

        if not resp.results:
            return STTResult(transcript="", language_code=self._normalize_lang(lang_codes[0]), confidence=0.0)

        result = resp.results[0]
        alt = result.alternatives[0]
        transcript = (alt.transcript or "").strip()
        confidence = float(getattr(alt, "confidence", 0.0) or 0.0)
        raw_lang = getattr(result, "language_code", None) or lang_codes[0]

        return STTResult(
            transcript=transcript,
            language_code=self._normalize_lang(raw_lang),
            confidence=confidence,
        )

    # streaming_recognize를 지원하지 않는 모델 → audio 수집 후 batch recognize
    _BATCH_ONLY_MODELS = {"long", "latest_long"}

    async def stream_events(
        self,
        audio_q: "asyncio.Queue[Optional[bytes]]",
        *,
        primary_language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        enable_punctuation: Optional[bool] = None,
        interim_results: bool = True,
        single_utterance: bool = False,  # v2에서 무시
        speech_phrases: Optional[List[str]] = None,
    ) -> AsyncIterator[STTStreamEvent]:
        lang_codes = self._resolve_language_codes(primary_language=primary_language)
        recog_cfg = self._build_recognition_config(
            language_codes=lang_codes,
            sample_rate_hz=sample_rate_hz,
            enable_punctuation=enable_punctuation,
            speech_phrases=speech_phrases,
        )

        # 실제 사용될 모델 확인 (long 폴백 여부)
        resolved_model, _ = self._resolve_model_and_langs(lang_codes)

        if resolved_model in self._BATCH_ONLY_MODELS:
            # long 모델: streaming 미지원 → audio 전체 수집 후 batch recognize
            async for ev in self._batch_stream_events(audio_q, recog_cfg, lang_codes):
                yield ev
            return

        streaming_cfg = cloud_speech.StreamingRecognitionConfig(
            config=recog_cfg,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=interim_results,
            ),
        )

        recognizer = self._recognizer
        out_q: "asyncio.Queue[Optional[object]]" = asyncio.Queue()
        # asyncio Queue → thread Queue 브릿지 (run_coroutine_threadsafe 없이 안전하게 전달)
        t_audio_q: "thread_queue.Queue[Optional[bytes]]" = thread_queue.Queue()

        async def _bridge_audio():
            """asyncio audio_q → thread t_audio_q 로 중계."""
            while True:
                chunk = await audio_q.get()
                t_audio_q.put(chunk)
                if chunk is None:
                    break

        loop = asyncio.get_running_loop()

        def _run_streaming_recognize():
            def req_gen():
                yield cloud_speech.StreamingRecognizeRequest(
                    recognizer=recognizer,
                    streaming_config=streaming_cfg,
                )
                while True:
                    try:
                        chunk = t_audio_q.get(timeout=30.0)  # 30초 무응답 시 스트림 종료
                    except thread_queue.Empty:
                        break
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
                        if not transcript:
                            continue
                        conf = float(getattr(alt, "confidence", 0.0) or 0.0)
                        raw_lang = getattr(result, "language_code", None) or lang_codes[0]
                        event = STTStreamEvent(
                            transcript=transcript,
                            language_code=self._normalize_lang(raw_lang),
                            is_final=bool(result.is_final),
                            confidence=conf,
                        )
                        asyncio.run_coroutine_threadsafe(out_q.put(event), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(out_q.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(out_q.put(None), loop).result()

        bridge_task = asyncio.create_task(_bridge_audio())
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
            await asyncio.gather(bridge_task, worker_task, return_exceptions=True)

    async def _batch_stream_events(
        self,
        audio_q: "asyncio.Queue[Optional[bytes]]",
        recog_cfg: cloud_speech.RecognitionConfig,
        lang_codes: List[str],
    ) -> AsyncIterator[STTStreamEvent]:
        """audio_q를 소진한 뒤 batch recognize로 최종 결과 하나를 방출."""
        chunks: List[bytes] = []
        while True:
            chunk = await audio_q.get()
            if chunk is None:
                break
            chunks.append(chunk)

        audio_bytes = b"".join(chunks)
        if not audio_bytes:
            return

        request = cloud_speech.RecognizeRequest(
            recognizer=self._recognizer,
            config=recog_cfg,
            content=audio_bytes,
        )

        try:
            resp = await asyncio.to_thread(self.client.recognize, request=request)
        except Exception:
            return

        for result in resp.results:
            if not result.alternatives:
                continue
            alt = result.alternatives[0]
            transcript = (alt.transcript or "").strip()
            if not transcript:
                continue
            conf = float(getattr(alt, "confidence", 0.0) or 0.0)
            raw_lang = getattr(result, "language_code", None) or lang_codes[0]
            yield STTStreamEvent(
                transcript=transcript,
                language_code=self._normalize_lang(raw_lang),
                is_final=True,
                confidence=conf,
            )