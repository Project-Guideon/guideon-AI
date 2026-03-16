from __future__ import annotations

from app.graph.state import GraphState

_SIMILARITY_THRESHOLD = 0.2   # 최상위 청크 유사도가 이 값 미만이면 bad
_MIN_ANSWER_LENGTH = 10       # 답변이 이 글자 수 미만이면 bad
_MAX_RETRY = 1                # 최대 재시도 횟수 (1회: 일반검색 → MMR 재검색)
_TOP_K_RETRY = 10             # 재시도 시 top_k

# 답변이 이 문구를 포함하면 사실상 모른다는 뜻 → bad 판정
_BAD_ANSWER_SIGNALS = [
    "관련 정보를 찾을 수 없",
    "정보가 없",
    "I don't have",
    "I couldn't find",
    "no information",
    "not found",
    "没有相关",
    "見つかりません",
]


def answer_check_node(state: GraphState) -> dict:
    """답변 품질 판정 + 재시도 로직.

    check_result:
        "good"  → tts_builder 로 이동
        "retry" → retry_count 증가, top_k 확대 후 retrieve 재시도
        "bad"   → 최대 재시도 초과, clarify 노드로 fallback
    """
    chunks: list = state.get("retrieved_chunks") or []
    answer: str = state.get("answer_text", "")
    retry_count: int = state.get("retry_count", 0)

    # ── Bad 판정 ──────────────────────────────────────────────────────
    is_bad = False

    if not chunks:
        is_bad = True                                          # 청크 없음
    elif max(c.get("similarity", 0.0) for c in chunks) < _SIMILARITY_THRESHOLD:
        is_bad = True                                          # 유사도 임계값 미달
    elif len(answer.strip()) < _MIN_ANSWER_LENGTH:
        is_bad = True                                          # 답변 너무 짧음
    else:
        for signal in _BAD_ANSWER_SIGNALS:
            if signal.lower() in answer.lower():
                is_bad = True                                  # "모른다" 신호
                break

    # ── 결과 결정 ─────────────────────────────────────────────────────
    trace = dict(state.get("trace") or {})
    flow = list(trace.get("_flow") or [])
    flow.append("answer_check")
    trace["_flow"] = flow

    if not is_bad:
        trace["answer_check"] = {"result": "good", "retry_count": retry_count}
        return {"check_result": "good", "trace": trace}

    if retry_count < _MAX_RETRY:
        new_retry = retry_count + 1
        trace["answer_check"] = {
            "result": "retry",
            "retry_count": new_retry,
            "top_k": _TOP_K_RETRY,
        }
        return {
            "check_result": "retry",
            "retry_count": new_retry,
            "top_k": _TOP_K_RETRY,
            "trace": trace,
        }

    # 최대 재시도 초과 → clarify fallback
    trace["answer_check"] = {"result": "bad", "retry_count": retry_count}
    return {"check_result": "bad", "trace": trace}
