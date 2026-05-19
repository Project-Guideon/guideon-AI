from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LanguageProfile:
    """STT 언어 코드와 답변 언어를 분리 관리."""

    user_language: str
    stt_language_code: Optional[str]  # None 이면 STT 엔진이 자동 감지
    answer_language: str


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "ko":   LanguageProfile("ko", "ko",   "ko"),
    "en":   LanguageProfile("en", "en",   "en"),
    "ja":   LanguageProfile("ja", "ja",   "ja"),
    "zh":   LanguageProfile("zh", "zh",   "zh"),
    "auto": LanguageProfile("ko", None,   "ko"),  # STT 언어 미지정 → 텍스트 감지 후 answer_language 갱신
}

_DEFAULT_PROFILE = LANGUAGE_PROFILES["ko"]


_BCP47_OVERRIDES: dict[str, str] = {
    "cmn": "zh",  # cmn-Hans-CN, cmn-Hant-TW 등 → zh
}


def get_profile(lang: Optional[str]) -> LanguageProfile:
    """2자리 코드 또는 BCP-47 또는 'auto' → LanguageProfile.

    예: "en", "en-US", "en-GB" → en 프로파일.
        "cmn-Hans-CN", "cmn" → zh 프로파일.
        None, "", "auto"     → auto 프로파일 (stt_language_code=None).
    """
    if not lang or lang.lower() == "auto":
        return LANGUAGE_PROFILES["auto"]
    lang2 = lang.split("-")[0].lower()
    lang2 = _BCP47_OVERRIDES.get(lang2, lang2)
    return LANGUAGE_PROFILES.get(lang2, _DEFAULT_PROFILE)
