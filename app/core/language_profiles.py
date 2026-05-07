from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class LanguageProfile:
    """STT 언어 코드와 답변 언어를 분리 관리."""

    user_language: str      # 2자리: ko / en / ja / zh
    stt_language_code: str  # Google STT BCP-47 코드
    answer_language: str    # 답변 생성 언어 (기본적으로 user_language와 동일)


# 영어는 한국어 고유명사 인식 정확도를 위해 ko-KR STT 사용
LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "ko": LanguageProfile("ko", "ko-KR",       "ko"),
    "en": LanguageProfile("en", "ko-KR",       "en"),
    "ja": LanguageProfile("ja", "ja-JP",       "ja"),
    "zh": LanguageProfile("zh", "cmn-Hans-CN", "zh"),
}

_DEFAULT_PROFILE = LANGUAGE_PROFILES["ko"]


_BCP47_OVERRIDES: dict[str, str] = {
    "cmn": "zh",  # cmn-Hans-CN, cmn-Hant-TW 등 → zh
}


def get_profile(lang: str) -> LanguageProfile:
    """2자리 코드 또는 BCP-47 → LanguageProfile.

    예: "en", "en-US", "en-GB" → en 프로파일.
        "cmn-Hans-CN", "cmn" → zh 프로파일.
    """
    lang2 = (lang or "").split("-")[0].lower()
    lang2 = _BCP47_OVERRIDES.get(lang2, lang2)
    return LANGUAGE_PROFILES.get(lang2, _DEFAULT_PROFILE)


