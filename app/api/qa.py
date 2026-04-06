"""
QA 엔드포인트
- POST /internal/v1/qa : Spring Boot Core 연동 (intent_gate가 4개 라우트 자동 분류)
    - struct_db  : 위치 안내  — nearbyPlaces context 기반 장소 추천
    - event      : 운영정보  — dailyInfos context 기반 답변 (현재 stub → RAG fallback)
    - rag        : 지식 검색 — 벡터DB 검색 기반 답변
    - smalltalk  : 일상 대화 — LLM 직접 답변
- POST /voice_qa       : 음성 파일 → 답변 음성 반환
- POST /text_qa        : 텍스트 질문 → 텍스트 답변 반환
"""
import asyncio
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from app.core.dependencies import traced_voice_run, traced_text_run, text_pipeline

router = APIRouter()


# ── /internal/v1/qa 모델 (Spring Boot Core QaRequest/QaResponse 와 1:1 매핑) ──

class DeviceLocation(BaseModel):
    latitude: float
    longitude: float


class DailyInfoSummary(BaseModel):
    placeName: str
    infoType: str
    content: str


class MascotPromptConfig(BaseModel):
    base_persona: Optional[str] = None
    smalltalk_style: Optional[str] = None
    event_node_style: Optional[str] = None
    RAG_style: Optional[str] = None
    struct_db_style: Optional[str] = None


class QaContext(BaseModel):
    dailyInfos: List[DailyInfoSummary] = []


class InternalQaRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sessionId: str
    siteId: int
    deviceId: str
    question: str
    language: Optional[str] = "ko"
    systemPrompt: Optional[str] = None   # tb_mascot.system_prompt (마스코트 캐릭터 설정)
    deviceLocation: Optional[DeviceLocation] = None
    context: Optional[QaContext] = None  # dailyInfos 전달용
    # 마스코트 개성 필드 (Spring JSON "name" → Python .mascotName)
    mascotName: Optional[str] = Field(None, alias="name")
    greetingMsg: Optional[str] = None
    promptConfig: Optional[MascotPromptConfig] = None


class InternalQaResponse(BaseModel):
    answer: str
    placeId: Optional[int] = None
    emotion: str = "HAPPY"
    language: str = "ko"
    category: str = "GENERAL"
    answerFound: bool = True


@router.post("/internal/v1/qa", response_model=InternalQaResponse)
async def internal_qa(req: InternalQaRequest):
    """Spring Boot Core 에서 호출하는 메인 QA 엔드포인트.

    - intent_gate 가 질문을 분류해 4개 라우트 중 하나로 자동 분기
    - struct_db  : Core 가 조립한 nearbyPlaces 로 위치 안내
    - event      : Core 가 조립한 dailyInfos 로 운영정보 답변 (현재 stub)
    - rag        : 벡터DB 검색 기반 지식 답변
    - smalltalk  : LLM 직접 일상 대화 답변
    - QaResponse 형식으로 반환
    """
    lang2 = (req.language or "ko").split("-")[0].lower()

    daily_infos = []
    if req.context:
        daily_infos = [d.model_dump() for d in req.context.dailyInfos]

    initial_state = {
        "transcript": req.question,
        "language_code": lang2,
        "user_language": lang2,
        "site_id": req.siteId,
        "device_id": req.deviceId,
        "system_prompt": req.systemPrompt or "",
        "mascot_name":            req.mascotName or "",
        "mascot_greeting":        req.greetingMsg or "",
        "mascot_base_persona":    (req.promptConfig.base_persona or "") if req.promptConfig else "",
        "mascot_smalltalk_style": (req.promptConfig.smalltalk_style or "") if req.promptConfig else "",
        "mascot_struct_db_style": (req.promptConfig.struct_db_style or "") if req.promptConfig else "",
        "mascot_RAG_style":       (req.promptConfig.RAG_style or "") if req.promptConfig else "",
        "mascot_event_style":     (req.promptConfig.event_node_style or "") if req.promptConfig else "",
        "top_k": 5,
        "retry_count": 0,
        "trace": {},
        "nearby_places": [],        # fetch_places_node 가 채움
        "daily_infos": daily_infos,
        "place_category": None,
        # struct_db_node 결과를 받기 위해 초기화
        "place_id": None,
        "emotion": "",
        "category": "",
    }

    result = await asyncio.to_thread(text_pipeline.graph.invoke, initial_state)

    answer = result.get("answer_text", "")
    answer_found = result.get("check_result") == "good"

    fallback_answers = {
        "ko": "죄송합니다, 해당 정보를 찾을 수 없습니다.",
        "en": "Sorry, I couldn't find that information.",
        "ja": "申し訳ありませんが、その情報は見つかりませんでした。",
        "zh": "抱歉, 我没有找到相关信息。",
    }

    return InternalQaResponse(
        answer=answer if answer_found else fallback_answers.get(lang2, fallback_answers["en"]),
        placeId=result.get("place_id"),
        emotion=result.get("emotion") or ("GUIDING" if answer_found else "SORRY"),
        language=lang2,
        category=result.get("category") or "GENERAL",
        answerFound=answer_found,
    )


@router.post("/voice_qa")
async def voice_qa(
    audio: UploadFile = File(...),
    site_id: int = Form(1),
    # 테스트용 mascot 필드 (internal/v1/qa와 동일한 구조, Form으로 전달)
    system_prompt: Optional[str] = Form(None),
    mascot_name: Optional[str] = Form(None),
    mascot_greeting: Optional[str] = Form(None),
    mascot_base_persona: Optional[str] = Form(None),
    mascot_smalltalk_style: Optional[str] = Form(None),
    mascot_struct_db_style: Optional[str] = Form(None),
    mascot_RAG_style: Optional[str] = Form(None),
    mascot_event_style: Optional[str] = Form(None),
):
    audio_bytes = await audio.read()
    mascot = _build_mascot_dict(
        system_prompt=system_prompt,
        mascot_name=mascot_name,
        mascot_greeting=mascot_greeting,
        prompt_config=MascotPromptConfig(
            base_persona=mascot_base_persona,
            smalltalk_style=mascot_smalltalk_style,
            struct_db_style=mascot_struct_db_style,
            RAG_style=mascot_RAG_style,
            event_node_style=mascot_event_style,
        ),
    )
    result = await asyncio.to_thread(traced_voice_run, audio_bytes, site_id, mascot)
    return Response(
        content=result.voice_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=answer.mp3"},
    )


class TextQARequest(BaseModel):
    query: str
    site_id: int = 1
    language_code: str = "ko-KR"
    # 테스트용 mascot 필드 (internal/v1/qa와 동일한 구조)
    system_prompt: Optional[str] = None
    mascot_name: Optional[str] = None
    mascot_greeting: Optional[str] = None
    prompt_config: Optional[MascotPromptConfig] = None


def _build_mascot_dict(
    system_prompt: Optional[str],
    mascot_name: Optional[str],
    mascot_greeting: Optional[str],
    prompt_config: Optional[MascotPromptConfig],
) -> dict:
    """mascot 관련 필드를 GraphState 키 형태로 변환."""
    pc = prompt_config
    return {
        "system_prompt":          system_prompt or "",
        "mascot_name":            mascot_name or "",
        "mascot_greeting":        mascot_greeting or "",
        "mascot_base_persona":    (pc.base_persona or "") if pc else "",
        "mascot_smalltalk_style": (pc.smalltalk_style or "") if pc else "",
        "mascot_struct_db_style": (pc.struct_db_style or "") if pc else "",
        "mascot_RAG_style":       (pc.RAG_style or "") if pc else "",
        "mascot_event_style":     (pc.event_node_style or "") if pc else "",
    }


@router.post("/text_qa")
async def text_qa(req: TextQARequest):
    mascot = _build_mascot_dict(
        req.system_prompt, req.mascot_name, req.mascot_greeting, req.prompt_config
    )
    result = await asyncio.to_thread(
        traced_text_run, req.query, req.site_id, req.language_code, mascot
    )
    return JSONResponse({"query": result.query, "answer": result.answer})
