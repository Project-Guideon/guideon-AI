from __future__ import annotations

import re

from app.core.services.tts_google import GoogleTTS, TTSConfig
from app.graph.state import GraphState

# 언어별 Google TTS language_code 매핑
_TTS_LANG_MAP = {
    "ko": "ko-KR",
    "en": "en-US",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "fr": "fr-FR",
    "es": "es-ES",
}


def _process_for_tts(text: str, lang: str = "ko") -> str:
    """답변 텍스트를 TTS 에 적합하게 가공.

    - 마크다운 기호 제거 (* _ ` # 등)
    - 연속 공백 정리
    - 한국어 전용: 괄호 안 내용 제거 (음성에서 불필요)
    """
    # 마크다운 기호 제거
    text = re.sub(r"[*_`#>]", "", text)

    # 한국어 전용 처리
    if lang == "ko":
        # 괄호 안 부연 설명 제거 (예: "(출처: ...)" 류)
        text = re.sub(r"\(출처[^)]*\)", "", text)
        text = re.sub(r"\[출처[^\]]*\]", "", text)

    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_tts_builder_node(tts: GoogleTTS):
    """TTS 텍스트 빌더 + Google TTS 호출 노드 팩토리.

    - answer_text 를 TTS 에 맞게 가공 → tts_text
    - user_language 에 맞는 TTS 설정으로 음성 합성
    - tts_audio(bytes) 를 state 에 저장
    """

    def tts_builder_node(state: GraphState) -> dict:
        answer_text: str = state.get("answer_text", "")
        user_language: str = state.get("user_language", "ko")

        if not answer_text.strip():
            answer_text = (
                "죄송합니다. 답변을 생성하지 못했습니다."
                if user_language == "ko"
                else "Sorry, I couldn't generate an answer."
            )

        tts_text = _process_for_tts(answer_text, user_language)

        # user_language 에 맞는 TTS 언어 코드 선택
        tts_lang = _TTS_LANG_MAP.get(user_language, "ko-KR")

        # 기본 TTS 가 ko-KR 이면 그대로, 다른 언어면 동적으로 새 인스턴스 생성
        if tts_lang == tts.config.language_code:
            active_tts = tts
        else:
            active_tts = GoogleTTS(TTSConfig(language_code=tts_lang))

        tts_audio = active_tts.synthesize(tts_text)

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("tts_builder")
        trace["_flow"] = flow
        trace["tts_builder"] = {
            "tts_lang": tts_lang,
            "tts_text_length": len(tts_text),
        }

        return {"tts_text": tts_text, "tts_audio": tts_audio, "trace": trace}

    return tts_builder_node
