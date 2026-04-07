from __future__ import annotations

import re
import unicodedata

from app.graph.state import GraphState


def normalize_node(state: GraphState) -> dict:
    """텍스트 정제 — 서비스 의존 없는 순수 로직."""
    text: str = state.get("transcript", "")

    # 1. Unicode NFC 정규화 (한글 자모 합성 등)
    text = unicodedata.normalize("NFC", text)

    # 2. 발화 잡음 제거: 문장 앞뒤 "음..." "어..." 류
    #    한글도 \w에 포함되므로, 한글·영숫자가 바로 앞뒤에 있으면 제거하지 않음
    text = re.sub(
        r"(?<![가-힣a-zA-Z0-9])(음+|어+|아+|에+|으+)(?![가-힣a-zA-Z0-9])",
        "",
        text,
    )

    # 3. 반복 문자 3개 초과 → 2개로 축약 (ㅋㅋㅋㅋ → ㅋㅋ)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 4. 연속 공백 → 단일 공백, 앞뒤 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    trace = dict(state.get("trace") or {})
    flow = list(trace.get("_flow") or [])
    flow.append("normalize")
    trace["_flow"] = flow
    trace["normalize"] = {"normalized_text": text}

    return {"normalized_text": text, "trace": trace}
