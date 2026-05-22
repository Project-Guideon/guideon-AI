"""
음성 클로닝 API

Cartesia voices.clone() 를 통해 관리자가 업로드한 음성 샘플로
커스텀 보이스를 생성하고, 생성된 voice_id를 반환합니다.
반환된 voice_id는 tb_mascot.tts_voice_id 에 저장되어 마스코트 TTS에 사용됩니다.
"""
from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.dependencies import cartesia_tts

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/voices", tags=["voices"])


class VoiceCloneResponse(BaseModel):
    voice_id: str
    name: str


@router.post("/clone", response_model=VoiceCloneResponse, summary="음성 클로닝")
async def clone_voice(
    file: UploadFile = File(..., description="클로닝에 사용할 음성 샘플 (WAV/MP3, 권장 10~30초)"),
    name: str = Form(..., description="생성할 보이스 이름 (마스코트 이름 등)"),
    language: str = Form(default="ko", description="ISO 639-1 언어 코드 (ko, en, ja, zh)"),
):
    """
    음성 샘플을 업로드하여 Cartesia 커스텀 보이스를 생성합니다.

    - 생성된 **voice_id** 를 마스코트 ttsVoiceId 필드에 저장하면 해당 음성으로 TTS가 동작합니다.
    - Cartesia API key가 설정되지 않은 경우 503을 반환합니다.
    """
    if cartesia_tts is None:
        raise HTTPException(
            status_code=503,
            detail="Cartesia TTS가 설정되지 않았습니다. CARTESIA_API_KEY 환경변수를 확인하세요.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어있습니다.")

    # 원본 확장자 보존 — Cartesia가 파일 형식을 자동 감지함
    original_name = file.filename or "voice_sample.wav"
    suffix = os.path.splitext(original_name)[1] or ".wav"

    tmp_path: str | None = None
    try:
        # 임시 파일에 저장 후 filepath 문자열로 전달 (voices.clone 은 경로를 받음)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # mode 기본값 "similarity" 사용 — clip은 voice_id 없이 임베딩만 반환하므로 불가
        voice_id, voice_name = await cartesia_tts.clone_voice_async(
            filepath=tmp_path,
            name=name,
            language=language,
        )

        logger.info("음성 클로닝 완료: voice_id=%s, name=%s", voice_id, voice_name)
        return VoiceCloneResponse(voice_id=voice_id, name=voice_name)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("음성 클로닝 실패: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"음성 클로닝 실패: {exc}") from exc

    finally:
        # 임시 파일 정리
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
