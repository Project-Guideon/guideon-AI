from __future__ import annotations

import json
import re
from typing import Optional

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

# ── 키워드 룰 ─────────────────────────────────────────────────────────────────
# 우선순위: map_tool > struct_db > rag > direct_llm

_MAP_TOOL_RULES = [
    r"화장실|주차|주차장|출입구|입구|출구|길찾기|가는 법|어디있|어디 있",
    r"근처|가까운|층|건물|몇 층|몇층",
    r"where is|restroom|toilet|parking|exit|entrance|directions|nearby|how to get",
    r"厕所|停车场|在哪|出口|入口|怎么走",
]

_STRUCT_DB_RULES = [
    r"운영시간|개장|폐장|영업시간|몇 시|몇시|오픈|클로즈|쉬는 날|휴무",
    r"요금|입장료|가격|얼마|비용|티켓|무료|유료",
    r"예약|신청|접수|문의|연락처|전화번호|이메일",
    r"규정|규칙|금지|반입|제한|드레스코드",
    r"opening.?hours|closing.?time|admission|price|fee|reservation|contact|rules|policy",
    r"开放时间|价格|预约|联系|规定",
]

_RAG_RULES = [
    r"역사|유래|의미|상징|이야기|전설|문화재|유물|인물|배경|특징|특색",
    r"설명|소개|어떤 곳|어떤곳|어떻게 만들|지어졌|창건|건립|설립|세워",
    r"history|origin|explain|describe|story|legend|meaning|symbol|heritage|background",
    r"历史|起源|介绍|故事|传说|意义|背景",
]

_DIRECT_LLM_RULES = [
    r"추천|코스|일정|팁|제안|뭐가 좋|어떤.*좋|정리|요약|전반적",
    r"recommend|suggest|tips|course|summary|generally|best",
    r"推荐|建议|总结",
]


def _rule_infotype(text: str) -> Optional[str]:
    for pattern in _MAP_TOOL_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return "map_tool"
    for pattern in _STRUCT_DB_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return "struct_db"
    for pattern in _RAG_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return "rag"
    for pattern in _DIRECT_LLM_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return "direct_llm"
    return None  # 애매 → LLM fallback


# ── 팩토리 ─────────────────────────────────────────────────────────────────────

def make_infotype_gate_node(llm: OpenAILLM):
    """LLM 서비스를 주입받아 노드 함수를 반환하는 팩토리."""

    def infotype_gate_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")

        info_type = _rule_infotype(text)
        method = "rule"

        if info_type is None:
            # ── LLM fallback (분류만, 답변 생성 절대 금지) ──
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an information-type classifier for a tourism voice chatbot.\n"
                        "Classify the query into EXACTLY one of: map_tool | struct_db | rag | direct_llm\n"
                        "  map_tool   : location/direction (bathroom, parking, exit, floor, nearby)\n"
                        "  struct_db  : operational info (hours, price, reservation, contact, rules)\n"
                        "  rag        : knowledge/history/explanation (history, origin, stories, descriptions)\n"
                        "  direct_llm : general advice/recommendation, no documents needed\n"
                        "When in doubt, prefer rag.\n"
                        "Respond ONLY with valid JSON. Example: {\"info_type\": \"rag\"}\n"
                        "DO NOT generate any answer or explanation."
                    ),
                },
                {"role": "user", "content": f"Classify this query: {text}"},
            ]
            try:
                raw = llm.chat(messages, max_tokens=30)
                info_type = json.loads(raw).get("info_type", "rag")
                method = "llm"
            except Exception:
                info_type = "rag"  # 파싱 실패 시 안전한 기본값
                method = "llm_fallback_default"

        trace = dict(state.get("trace") or {})
        trace["infotype_gate"] = {"text": text, "info_type": info_type, "method": method}

        return {"info_type": info_type, "trace": trace}

    return infotype_gate_node
