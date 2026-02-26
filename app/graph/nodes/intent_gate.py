from __future__ import annotations

import json
import re
from typing import Optional

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

# ── 키워드 룰 ─────────────────────────────────────────────────────────────────
# info_request를 먼저 검사 — 정보 요청이 더 중요하고 명확하게 분류 가능

_INFO_REQUEST_RULES = [
    r"어디|어떻게|언제|누가|누구|무엇|알려|설명|말해|가르쳐|뭐야|어때",
    r"입장료|요금|가격|얼마|비용|티켓|무료|유료",
    r"운영시간|개장|폐장|영업|휴무|쉬는날|오픈|클로즈",
    r"위치|장소|어디있|찾아가|가는방법|경로|방향",
    r"예약|신청|접수|문의|연락|전화|이메일",
    r"주차|화장실|식당|카페|편의",
    r"역사|유래|의미|상징|이야기|전설|문화재|유물|인물|배경|특징",
    r"where|when|what|how|who|history|explain|tell me|information",
    r"在哪|什么时候|怎么|介绍|历史|在哪里",
]

_SMALLTALK_RULES = [
    r"안녕|반가워|반갑|잘 있|잘있|좋아|싫어|힘들|기분|감정|느낌",
    r"재미|웃|농담|장난|심심|외로|배고|피곤|졸려",
    r"이름이 뭐|누구야|뭐하|어떻게 지내|잘 지내|잘지내",
    r"고마워|감사합|미안|죄송|수고|칭찬",
    r"hello|hi\b|hey\b|thanks|thank you|how are you|what.?s your name",
    r"你好|谢谢|叫什么|名字是",
]


def _rule_intent(text: str) -> Optional[str]:
    # info_request 먼저 검사 — 명확한 정보 요청 키워드가 있으면 즉시 반환
    for pattern in _INFO_REQUEST_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return "info_request"
    # 그 다음 smalltalk 검사
    for pattern in _SMALLTALK_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return "smalltalk"
    return None  # 둘 다 해당 없으면 LLM fallback


# ── 팩토리 ─────────────────────────────────────────────────────────────────────

def make_intent_gate_node(llm: OpenAILLM):
    """LLM 서비스를 주입받아 노드 함수를 반환하는 팩토리."""

    def intent_gate_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")

        intent = _rule_intent(text)
        method = "rule"

        if intent is None:
            # ── LLM fallback (분류만, 답변 생성 절대 금지) ──
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier for a tourism voice chatbot.\n"
                        "Classify the user input as EXACTLY one of: smalltalk | info_request\n"
                        "  info_request: questions about location, time, price, history, facilities, rules, etc.\n"
                        "  smalltalk   : greetings, emotional chat, self-introduction, jokes, casual conversation\n"
                        "When in doubt, prefer info_request.\n"
                        "Respond ONLY with valid JSON. Example: {\"intent\": \"info_request\"}\n"
                        "DO NOT generate any answer or explanation."
                    ),
                },
                {"role": "user", "content": f"Classify this input: {text}"},
            ]
            try:
                raw = llm.chat(messages, max_tokens=20)
                intent = json.loads(raw).get("intent", "info_request")
                method = "llm"
            except Exception:
                intent = "info_request"  # 파싱 실패 시 안전한 기본값
                method = "llm_fallback_default"

        trace = dict(state.get("trace") or {})
        trace["intent_gate"] = {"text": text, "intent": intent, "method": method}

        return {"intent": intent, "trace": trace}

    return intent_gate_node
