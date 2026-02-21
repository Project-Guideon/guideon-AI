from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_LANG_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese",
    "ja": "Japanese", "fr": "French", "es": "Spanish",
}


def make_answer_compose_node(llm: OpenAILLM):
    """map_tool / struct_db 결과를 자연어 문장으로 변환하는 노드 팩토리.

    stub 상태에서는 안내 데스크 문의 안내 문구를 반환.
    실제 API/DB 연결 후에는 결과 데이터를 자연어로 포맷팅.
    """

    def answer_compose_node(state: GraphState) -> dict:
        user_language: str = state.get("user_language", "ko")
        info_type: str = state.get("info_type", "map_tool")
        lang_name = _LANG_NAMES.get(user_language, user_language.upper())

        # map_tool / struct_db 결과 확인
        raw_result = state.get("poi_result") or state.get("db_result") or {}
        status = raw_result.get("status", "stub")
        data = raw_result.get("result")

        if status == "stub" or data is None:
            # ── stub: 안내 데스크 안내 문구 ──────────────────────────
            if user_language == "ko":
                if info_type == "map_tool":
                    answer = "해당 위치 정보 서비스가 준비 중입니다. 가까운 안내 데스크에 문의해 주시면 친절하게 안내해 드리겠습니다."
                else:
                    answer = "해당 정보 서비스가 준비 중입니다. 안내 데스크에 문의해 주시면 자세히 알려 드리겠습니다."
            else:
                if info_type == "map_tool":
                    answer = "The location service is currently being prepared. Please ask at the nearest information desk for assistance."
                else:
                    answer = "This information service is currently being prepared. Please contact the information desk for details."
        else:
            # ── 실제 데이터 → LLM 으로 자연어 변환 ──────────────────
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a tourism guide assistant. "
                        f"Convert the following structured data into a natural, speech-friendly response in {lang_name}. "
                        f"Keep it 1-3 sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Data: {data}\nUser question: {state.get('normalized_text', '')}",
                },
            ]
            try:
                answer = llm.chat(messages, max_tokens=150)
            except Exception:
                answer = (
                    "안내 데스크에 문의해 주세요."
                    if user_language == "ko"
                    else "Please contact the information desk."
                )

        trace = dict(state.get("trace") or {})
        trace["answer_compose"] = {
            "info_type": info_type,
            "status": status,
            "user_language": user_language,
        }

        return {"answer_text": answer, "trace": trace}

    return answer_compose_node
